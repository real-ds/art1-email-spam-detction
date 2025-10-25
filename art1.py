import numpy as np
import re
from typing import List, Tuple, Dict
import json

class ART1:
    """
    ART1 (Adaptive Resonance Theory 1) Neural Network for Binary Pattern Classification
    """
    
    def __init__(self, input_size: int, vigilance: float = 0.75, max_clusters: int = 100):
        """
        Initialize ART1 network
        
        Args:
            input_size: Number of binary input features
            vigilance: Vigilance parameter (0-1), controls cluster similarity threshold
            max_clusters: Maximum number of clusters to create
        """
        self.input_size = input_size
        self.vigilance = vigilance
        self.max_clusters = max_clusters
        self.n_clusters = 0
        
        # Bottom-up weights (F1 to F2 layer)
        self.weights_bu = np.ones((max_clusters, input_size)) / (1 + input_size)
        
        # Top-down weights (F2 to F1 layer)
        self.weights_td = np.ones((max_clusters, input_size))
        
        # Cluster labels (for supervised learning)
        self.cluster_labels = {}
        
        # Statistics
        self.cluster_sizes = np.zeros(max_clusters)
        
    def _compute_choice_function(self, input_vector: np.ndarray) -> np.ndarray:
        """Compute choice function for all clusters"""
        if self.n_clusters == 0:
             return np.array([])
        choices = np.sum(self.weights_bu[:self.n_clusters] * input_vector, axis=1) / (
            0.5 + np.sum(self.weights_bu[:self.n_clusters], axis=1)
        )
        return choices
    
    def _check_vigilance(self, input_vector: np.ndarray, cluster_idx: int) -> bool:
        """Check if input meets vigilance criterion for given cluster"""
        intersection = np.logical_and(input_vector, self.weights_td[cluster_idx])
        match_ratio = np.sum(intersection) / np.sum(input_vector)
        return match_ratio >= self.vigilance
    
    def _create_new_cluster(self, input_vector: np.ndarray) -> int:
        """Create a new cluster for the input pattern"""
        if self.n_clusters >= self.max_clusters:
            return -1
        
        cluster_idx = self.n_clusters
        self.weights_bu[cluster_idx] = input_vector / (0.5 + np.sum(input_vector))
        self.weights_td[cluster_idx] = input_vector
        self.n_clusters += 1
        self.cluster_sizes[cluster_idx] = 1
        
        return cluster_idx
    
    def _update_cluster(self, input_vector: np.ndarray, cluster_idx: int):
        """Update cluster weights with new input"""
        # Update top-down weights (intersection)
        self.weights_td[cluster_idx] = np.logical_and(
            self.weights_td[cluster_idx], input_vector
        ).astype(float)
        
        # Update bottom-up weights
        self.weights_bu[cluster_idx] = self.weights_td[cluster_idx] / (
            0.5 + np.sum(self.weights_td[cluster_idx])
        )
        
        self.cluster_sizes[cluster_idx] += 1
    
    def train(self, input_vector: np.ndarray, label: str = None) -> int:
        """
        Train the network with a single input vector
        
        Args:
            input_vector: Binary input vector
            label: Optional label for supervised learning
            
        Returns:
            Assigned cluster index
        """
        input_vector = np.array(input_vector, dtype=float)
        
        if np.sum(input_vector) == 0:
            return -1
        
        # If no clusters exist, create first one
        if self.n_clusters == 0:
            cluster_idx = self._create_new_cluster(input_vector)
            if label:
                self.cluster_labels[cluster_idx] = label
            return cluster_idx
        
        # Find best matching cluster
        choices = self._compute_choice_function(input_vector)
        sorted_indices = np.argsort(choices)[::-1]
        
        for cluster_idx in sorted_indices:
            if self._check_vigilance(input_vector, cluster_idx):
                self._update_cluster(input_vector, cluster_idx)
                
                # Update label if provided
                if label:
                    if cluster_idx not in self.cluster_labels:
                        self.cluster_labels[cluster_idx] = label
                
                return cluster_idx
        
        # No matching cluster found, create new one
        cluster_idx = self._create_new_cluster(input_vector)
        if cluster_idx >= 0 and label:
            self.cluster_labels[cluster_idx] = label
            
        return cluster_idx
    
    def predict(self, input_vector: np.ndarray) -> Tuple[int, str]:
        """
        Predict cluster and label for input vector
        
        Returns:
            Tuple of (cluster_index, label)
        """
        input_vector = np.array(input_vector, dtype=float)
        
        if np.sum(input_vector) == 0 or self.n_clusters == 0:
            return -1, "unknown"
        
        # Find best matching cluster
        choices = self._compute_choice_function(input_vector)
        sorted_indices = np.argsort(choices)[::-1]
        
        for cluster_idx in sorted_indices:
            if self._check_vigilance(input_vector, cluster_idx):
                label = self.cluster_labels.get(cluster_idx, "unknown")
                return cluster_idx, label
        
        return -1, "unknown"
    
    def get_statistics(self) -> Dict:
        """Get network statistics"""
        return {
            "n_clusters": self.n_clusters,
            "cluster_sizes": self.cluster_sizes[:self.n_clusters].tolist(),
            "cluster_labels": self.cluster_labels,
            "vigilance": self.vigilance
        }


class EmailFeatureExtractor:
    """Extract binary features from email text"""
    
    def __init__(self):
        # Spam indicators (19 features)
        self.spam_keywords = [
            'free', 'winner', 'cash', 'prize', 'click here', 'buy now',
            'limited time', 'act now', 'urgent', 'congratulations',
            'viagra', 'pharmacy', 'loan', 'credit', 'investment',
            'million', 'guarantee', 'risk free', 'no obligation'
        ]
        
        # Legitimate indicators (9 features)
        self.legit_keywords = [
            'meeting', 'schedule', 'report', 'update', 'team',
            'project', 'deadline', 'attached', 'please review'
        ]
        
    def extract_features(self, email_text: str, subject: str = "") -> np.ndarray:
        """
        Extract binary features from email
        
        Returns:
            Binary feature vector
        """
        text = (email_text + " " + subject).lower()
        subj_lower = subject.lower()
        features = []
        
        # Spam keyword features (19 features)
        for keyword in self.spam_keywords:
            features.append(1 if keyword in text else 0)
        
        # Legitimate keyword features (9 features)
        for keyword in self.legit_keywords:
            features.append(1 if keyword in text else 0)
        
        # --- FIXED SECTION ---
        # Pattern features (12 features to make total 40)
        # 19 + 9 = 28. 28 + 12 = 40.
        features.append(1 if text.count('!') > 2 else 0)  # 1: Multiple exclamation
        features.append(1 if text.count('?') > 2 else 0)  # 2: Multiple question marks (FIXED)
        features.append(1 if subject.isupper() and subject != "" else 0)  # 3: All caps subject
        features.append(1 if 'http' in text else 0)  # 4: Contains a link
        features.append(1 if re.search(r'\d{4,}', text) else 0)  # 5: Contains 4+ digits together
        features.append(1 if '$' in text or '€' in text or '£' in text else 0)  # 6: Currency symbols
        features.append(1 if 'account' in text else 0) # 7: 'account'
        features.append(1 if 'password' in text else 0) # 8: 'password'
        features.append(1 if 'verify' in text else 0) # 9: 'verify'
        features.append(1 if 're:' in subj_lower else 0) # 10: 're:' in subject
        features.append(1 if len(re.findall(r'[A-Z]', subject)) > len(subject) * 0.7 and len(subject) > 5 else 0) # 11: Subject > 70% caps
        features.append(1 if 'unsubscribe' in text else 0) # 12: 'unsubscribe'
        # --- END FIXED SECTION ---

        return np.array(features)
    
    def get_feature_size(self) -> int:
        """Get total number of features"""
        # 19 spam + 9 legit + 12 pattern = 40
        return 40


class SpamDetector:
    """Email spam detector using ART1"""
    
    def __init__(self, vigilance: float = 0.75):
        self.feature_extractor = EmailFeatureExtractor()
        self.art1 = ART1(
            input_size=self.feature_extractor.get_feature_size(),
            vigilance=vigilance,
            max_clusters=50
        )
        
    def train(self, email_text: str, subject: str = "", is_spam: bool = False):
        """Train with a single email"""
        features = self.feature_extractor.extract_features(email_text, subject)
        label = "spam" if is_spam else "ham"
        cluster_idx = self.art1.train(features, label)
        return cluster_idx
    
    def predict(self, email_text: str, subject: str = "") -> Tuple[str, int, float]:
        """
        Predict if email is spam
        
        Returns:
            Tuple of (prediction, cluster_index, confidence)
        """
        features = self.feature_extractor.extract_features(email_text, subject)
        cluster_idx, label = self.art1.predict(features)
        
        # Calculate simple confidence based on cluster size
        if cluster_idx >= 0:
            cluster_size = self.art1.cluster_sizes[cluster_idx]
            confidence = min(cluster_size / 10.0, 1.0)  # Normalize to 0-1
        else:
            confidence = 0.0
            label = "unknown"
        
        return label, cluster_idx, confidence
    
    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        return self.art1.get_statistics()


# Demo and Testing
if __name__ == "__main__":
    print("=" * 60)
    print("ART1 Email Spam Detector Demo")
    print("=" * 60)
    
    # Initialize detector
    detector = SpamDetector(vigilance=0.70)
    
    # Training data - Spam examples
    spam_emails = [
        ("Congratulations! You've won $1,000,000! Click here now!", "You're a WINNER!"),
        ("FREE VIAGRA! Buy now with 50% discount. Limited time offer!", "Amazing deal inside"),
        ("Get rich quick! Investment opportunity. Act now!", "Make money fast"),
        ("You have been selected for a cash prize. Click here to claim.", "Prize notification"),
        ("Urgent! Your account needs verification. Click this link immediately.", "Account Alert"),
        ("Buy cheap pharmacy products online. No prescription needed!", "Health products"),
        ("Work from home and earn $5000 per week! Risk free guarantee!", "Job opportunity"),
        ("Lose weight fast with our miracle pill! 100% guaranteed results!", "Lose 20 lbs now"),
    ]
    
    # Training data - Legitimate examples
    ham_emails = [
        ("Hi team, please find the project report attached. Let me know if you have questions.", "Project Update"),
        ("Meeting scheduled for tomorrow at 2 PM. Please review the agenda.", "Team Meeting"),
        ("The deadline for the quarterly report is next Friday. Thanks!", "Deadline Reminder"),
        ("Can you please review the attached document and provide feedback?", "Document Review"),
        ("Thanks for your help with the presentation. It went well!", "Thank you"),
        ("Please update the project status by end of day. Let me know if you need help.", "Status Update"),
        ("The team lunch is scheduled for Thursday at noon. See you there!", "Team Lunch"),
        ("I've attached the meeting notes from yesterday. Please review.", "Meeting Notes"),
    ]
    
    print("\n--- TRAINING PHASE ---")
    print("Training with spam emails...")
    for email, subject in spam_emails:
        cluster = detector.train(email, subject, is_spam=True)
        print(f"  Spam email assigned to cluster: {cluster}")
    
    print("\nTraining with legitimate emails...")
    for email, subject in ham_emails:
        cluster = detector.train(email, subject, is_spam=False)
        print(f"  Ham email assigned to cluster: {cluster}")
    
    # Display statistics
    print("\n--- NETWORK STATISTICS ---")
    stats = detector.get_statistics()
    print(f"Total clusters created: {stats['n_clusters']}")
    print(f"Vigilance parameter: {stats['vigilance']}")
    print(f"Cluster labels: {stats['cluster_labels']}")
    
    # Testing phase
    print("\n--- TESTING PHASE ---")
    test_emails = [
        ("FREE money! Click here to claim your prize now!!!", "Amazing offer", True),
        ("Can we schedule a meeting for next week to discuss the project?", "Meeting request", False),
        ("You won the lottery! Send us your bank details immediately!", "Lottery winner", True),
        ("Please find attached the quarterly financial report.", "Q4 Report", False),
        ("Buy cheap medications online! No questions asked!", "Pharmacy deals", True),
        ("The project deadline has been extended to next month.", "Deadline update", False),
    ]
    
    correct = 0
    total = len(test_emails)
    
    for email, subject, expected_spam in test_emails:
        prediction, cluster, confidence = detector.predict(email, subject)
        is_spam = (prediction == "spam")
        correct += (is_spam == expected_spam)
        
        status = "✓" if is_spam == expected_spam else "✗"
        print(f"\n{status} Email: {email[:50]}...")
        print(f"  Subject: {subject}")
        print(f"  Prediction: {prediction} (cluster {cluster}, confidence: {confidence:.2f})")
        print(f"  Expected: {'spam' if expected_spam else 'ham'}")
    
    accuracy = (correct / total) * 100
    print(f"\n--- RESULTS ---")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
    
    # Test with novel spam pattern
    print("\n--- TESTING NOVEL SPAM PATTERN ---")
    novel_spam = "Cryptocurrency investment! Double your Bitcoin in 24 hours! Limited spots available!"
    prediction, cluster, confidence = detector.predict(novel_spam, "Crypto opportunity")
    print(f"Novel spam email: {novel_spam}")
    print(f"Prediction: {prediction} (cluster {cluster}, confidence: {confidence:.2f})")
    
    # Train with the novel pattern
    print("\nTraining with novel spam pattern...")
    new_cluster = detector.train(novel_spam, "Crypto opportunity", is_spam=True)
    print(f"New cluster created: {new_cluster}")
    
    # Test again
    prediction, cluster, confidence = detector.predict(novel_spam, "Crypto opportunity")
    print(f"After training - Prediction: {prediction} (cluster {cluster}, confidence: {confidence:.2f})")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)