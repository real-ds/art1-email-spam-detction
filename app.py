import numpy as np
import re
from typing import List, Tuple, Dict
import json
from flask import Flask, render_template, request

# --- ART-1 and Email Classes (No changes here) ---
class ART1:
    """
    ART1 (Adaptive Resonance Theory 1) Neural Network for Binary Pattern Classification
    """
    
    def __init__(self, input_size: int, vigilance: float = 0.75, max_clusters: int = 100):
        self.input_size = input_size
        self.vigilance = vigilance
        self.max_clusters = max_clusters
        self.n_clusters = 0
        self.weights_bu = np.ones((max_clusters, input_size)) / (1 + input_size)
        self.weights_td = np.ones((max_clusters, input_size))
        self.cluster_labels = {}
        self.cluster_sizes = np.zeros(max_clusters)
        
    def _compute_choice_function(self, input_vector: np.ndarray) -> np.ndarray:
        if self.n_clusters == 0:
            return np.array([])
        choices = np.sum(self.weights_bu[:self.n_clusters] * input_vector, axis=1) / (
            0.5 + np.sum(self.weights_bu[:self.n_clusters], axis=1)
        )
        return choices
    
    def _check_vigilance(self, input_vector: np.ndarray, cluster_idx: int) -> bool:
        intersection = np.logical_and(input_vector, self.weights_td[cluster_idx])
        input_sum = np.sum(input_vector)
        if input_sum == 0:
            return False 
        match_ratio = np.sum(intersection) / input_sum
        return match_ratio >= self.vigilance
    
    def _create_new_cluster(self, input_vector: np.ndarray) -> int:
        if self.n_clusters >= self.max_clusters:
            return -1
        cluster_idx = self.n_clusters
        self.weights_bu[cluster_idx] = input_vector / (0.5 + np.sum(input_vector))
        self.weights_td[cluster_idx] = input_vector
        self.n_clusters += 1
        self.cluster_sizes[cluster_idx] = 1
        return cluster_idx
    
    def _update_cluster(self, input_vector: np.ndarray, cluster_idx: int):
        self.weights_td[cluster_idx] = np.logical_and(
            self.weights_td[cluster_idx], input_vector
        ).astype(float)
        self.weights_bu[cluster_idx] = self.weights_td[cluster_idx] / (
            0.5 + np.sum(self.weights_td[cluster_idx])
        )
        self.cluster_sizes[cluster_idx] += 1
    
    def train(self, input_vector: np.ndarray, label: str = None) -> int:
        input_vector = np.array(input_vector, dtype=float)
        if np.sum(input_vector) == 0:
            return -1
        if self.n_clusters == 0:
            cluster_idx = self._create_new_cluster(input_vector)
            if label:
                self.cluster_labels[cluster_idx] = label
            return cluster_idx
        choices = self._compute_choice_function(input_vector)
        sorted_indices = np.argsort(choices)[::-1]
        for cluster_idx in sorted_indices:
            if self._check_vigilance(input_vector, cluster_idx):
                self._update_cluster(input_vector, cluster_idx)
                if label:
                    if cluster_idx not in self.cluster_labels:
                        self.cluster_labels[cluster_idx] = label
                return cluster_idx
        cluster_idx = self._create_new_cluster(input_vector)
        if cluster_idx >= 0 and label:
            self.cluster_labels[cluster_idx] = label
        return cluster_idx
    
    def predict(self, input_vector: np.ndarray) -> Tuple[int, str]:
        input_vector = np.array(input_vector, dtype=float)
        if np.sum(input_vector) == 0 or self.n_clusters == 0:
            return -1, "unknown"
        choices = self._compute_choice_function(input_vector)
        sorted_indices = np.argsort(choices)[::-1]
        for cluster_idx in sorted_indices:
            if self._check_vigilance(input_vector, cluster_idx):
                label = self.cluster_labels.get(cluster_idx, "unknown")
                return cluster_idx, label
        return -1, "unknown"
    
    def get_statistics(self) -> Dict:
        return {
            "n_clusters": self.n_clusters,
            "cluster_sizes": self.cluster_sizes[:self.n_clusters].tolist(),
            "cluster_labels": self.cluster_labels,
            "vigilance": self.vigilance
        }

class EmailFeatureExtractor:
    def __init__(self):
        self.spam_keywords = [
            'free', 'winner', 'cash', 'prize', 'click here', 'buy now',
            'limited time', 'act now', 'urgent', 'congratulations',
            'viagra', 'pharmacy', 'loan', 'credit', 'investment',
            'million', 'guarantee', 'risk free', 'no obligation'
        ]
        self.legit_keywords = [
            'meeting', 'schedule', 'report', 'update', 'team',
            'project', 'deadline', 'attached', 'please review'
        ]
    def extract_features(self, email_text: str, subject: str = "") -> np.ndarray:
        text = (email_text + " " + subject).lower()
        subj_lower = subject.lower()
        features = []
        for keyword in self.spam_keywords: features.append(1 if keyword in text else 0)
        for keyword in self.legit_keywords: features.append(1 if keyword in text else 0)
        features.append(1 if text.count('!') > 2 else 0)
        features.append(1 if text.count('?') > 2 else 0)
        features.append(1 if subject.isupper() and subject != "" else 0)
        features.append(1 if 'http' in text else 0)
        features.append(1 if re.search(r'\d{4,}', text) else 0)
        features.append(1 if '$' in text or '€' in text or '£' in text else 0)
        features.append(1 if 'account' in text else 0)
        features.append(1 if 'password' in text else 0)
        features.append(1 if 'verify' in text else 0)
        features.append(1 if 're:' in subj_lower else 0)
        features.append(1 if len(re.findall(r'[A-Z]', subject)) > len(subject) * 0.7 and len(subject) > 5 else 0)
        features.append(1 if 'unsubscribe' in text else 0)
        return np.array(features)
    
    def get_feature_size(self) -> int:
        return 40

class SpamDetector:
    def __init__(self, vigilance: float = 0.75):
        self.feature_extractor = EmailFeatureExtractor()
        self.art1 = ART1(
            input_size=self.feature_extractor.get_feature_size(),
            vigilance=vigilance,
            max_clusters=50
        )
    def train(self, email_text: str, subject: str = "", is_spam: bool = False):
        features = self.feature_extractor.extract_features(email_text, subject)
        label = "spam" if is_spam else "ham"
        cluster_idx = self.art1.train(features, label)
        return cluster_idx
    
    def predict(self, email_text: str, subject: str = "") -> Tuple[str, int, float]:
        features = self.feature_extractor.extract_features(email_text, subject)
        cluster_idx, label = self.art1.predict(features)
        if cluster_idx >= 0:
            cluster_size = self.art1.cluster_sizes[cluster_idx]
            confidence = min(cluster_size / 10.0, 1.0)
        else:
            confidence = 0.0
            label = "unknown"
        return label, cluster_idx, confidence
    
    def get_statistics(self) -> Dict:
        return self.art1.get_statistics()
# --- End of Classes ---


# --- Data (Expanded) ---
spam_emails = [
    ("Congratulations! You've won $1,000,000! Click here now!", "You're a WINNER!"),
    ("FREE VIAGRA! Buy now with 50% discount. Limited time offer!", "Amazing deal inside"),
    ("Get rich quick! Investment opportunity. Act now!", "Make money fast"),
    ("You have been selected for a cash prize. Click here to claim.", "Prize notification"),
    ("Urgent! Your account needs verification. Click this link immediately.", "Account Alert"),
    ("Buy cheap pharmacy products online. No prescription needed!", "Health products"),
    ("Work from home and earn $5000 per week! Risk free guarantee!", "Job opportunity"),
    ("Lose weight fast with our miracle pill! 100% guaranteed results!", "Lose 20 lbs now"),
    # --- New Spam Data ---
    ("Your bank account has been temporarily suspended. Please click here to verify your account.", "Account Security Warning"),
    ("You won a free cruise! Claim your prize now, limited spots available.", "You are a Winner!"),
    ("Amazing low-interest loan offer just for you. Get cash now!", "Your Loan is Pre-approved"),
    ("Meet hot singles in your area! Click here to see profiles near you.", "New message received"),
    ("100% risk free investment! Double your money in 30 days. Guaranteed!", "Exclusive Investment Offer")
]

ham_emails = [
    ("Hi team, please find the project report attached. Let me know if you have questions.", "Project Update"),
    ("Meeting scheduled for tomorrow at 2 PM. Please review the agenda.", "Team Meeting"),
    ("The deadline for the quarterly report is next Friday. Thanks!", "Deadline Reminder"),
    ("Can you please review the attached document and provide feedback?", "Document Review"),
    ("Thanks for your help with the presentation. It went well!", "Thank you"),
    ("Please update the project status by end of day. Let me know if you need help.", "Status Update"),
    ("The team lunch is scheduled for Thursday at noon. See you there!", "Team Lunch"),
    ("I've attached the meeting notes from yesterday. Please review.", "Meeting Notes"),
    # --- New Ham Data ---
    ("Here are the slides from today's team presentation. Please review when you can.", "Today's Presentation Slides"),
    ("Reminder: The project deadline is this Friday. Let's sync up tomorrow.", "Project Sync"),
    ("Can you please send me the Q3 report? I need it for the update.", "Request for Q3 Report"),
    ("The team meeting has been rescheduled to 3 PM due to a conflict. See attached invite.", "Meeting Rescheduled"),
    ("Please find the attached invoice for last month's services.", "Invoice [INV-12345]")
]

test_emails = [
    ("FREE money! Click here to claim your prize now!!!", "Amazing offer", True),
    ("Can we schedule a meeting for next week to discuss the project?", "Meeting request", False),
    ("You won the lottery! Send us your bank details immediately!", "Lottery winner", True),
    ("Please find attached the quarterly financial report.", "Q4 Report", False),
    ("Buy cheap medications online! No questions asked!", "Pharmacy deals", True),
    ("The project deadline has been extended to next month.", "Deadline update", False),
    ("Cryptocurrency investment! Double your Bitcoin in 24 hours! Limited spots!", "Crypto opportunity", True),
    # --- New Test Data ---
    ("Your Amazon account shows a login from a new device. Click here to secure your account.", "Security Alert", True),
    ("Just following up on our meeting. Can you send the report?", "Follow-up", False),
    ("Get a 100% free credit report. No obligation. See your score now!", "Your Credit Score", True),
    ("The company picnic is next week. Please RSVP by EOD.", "Company Event", False),
    ("A package is waiting for you. Please confirm your shipping address to avoid delays.", "Your Package Delivery", True)
]

# --- Flask Application ---
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    vigilance_default = 0.70
    results_data = None
    graph_data = None 

    # Feature Explanation Data (always available)
    feature_extractor = EmailFeatureExtractor()
    spam_ex_text, spam_ex_subj = spam_emails[1] 
    spam_ex_vector = feature_extractor.extract_features(spam_ex_text, spam_ex_subj)
    spam_ex_vector_str = "".join(spam_ex_vector.astype(int).astype(str))
    spam_ex_vector_str = ' '.join(spam_ex_vector_str[i:i+10] for i in range(0, len(spam_ex_vector_str), 10))
    ham_ex_text, ham_ex_subj = ham_emails[1] 
    ham_ex_vector = feature_extractor.extract_features(ham_ex_text, ham_ex_subj)
    ham_ex_vector_str = "".join(ham_ex_vector.astype(int).astype(str))
    ham_ex_vector_str = ' '.join(ham_ex_vector_str[i:i+10] for i in range(0, len(ham_ex_vector_str), 10))
    
    explanation_data = [
        {"type": "spam", "subject": spam_ex_subj, "body": spam_ex_text, "vector": spam_ex_vector_str},
        {"type": "ham", "subject": ham_ex_subj, "body": ham_ex_text, "vector": ham_ex_vector_str}
    ]

    if request.method == 'POST':
        vigilance = float(request.form['vigilance'])
        vigilance_default = vigilance
        
        # 1. Re-initialize and Re-train
        detector = SpamDetector(vigilance=vigilance)
        for email, subject in spam_emails:
            detector.train(email, subject, is_spam=True)
        for email, subject in ham_emails:
            detector.train(email, subject, is_spam=False)
        stats = detector.get_statistics()
        
        # 2. Prepare Cluster Data for Visualization
        cluster_details = {}
        for cluster_id, label in stats['cluster_labels'].items():
            cluster_details[cluster_id] = {'id': cluster_id, 'label': label, 'emails': []}
        cluster_details[-1] = {'id': -1, 'label': 'unknown', 'emails': []}
        
        all_training_data = spam_emails + ham_emails
        vector_assignments = [] 
        
        for i, (email, subject) in enumerate(all_training_data):
            label, cluster_id, _ = detector.predict(email, subject)
            is_spam = i < len(spam_emails)
            vec_type = 'spam' if is_spam else 'ham'
            
            # --- UPDATED: Calculate spam/ham index correctly ---
            vec_name = f"Spam {i + 1}" if is_spam else f"Ham {i - len(spam_emails) + 1}"
            
            email_display = f"({vec_type}) {subject}"
            
            if cluster_id in cluster_details:
                 cluster_details[cluster_id]['emails'].append(email_display)
            
            # --- UPDATED: Add subject and body for the tooltip ---
            vector_assignments.append({
                'id': f"v_{i}", 
                'name': vec_name, 
                'type': vec_type, 
                'cluster_id': cluster_id,
                'subject': subject,
                'body': email
            })

        clusters_list = sorted(cluster_details.values(), key=lambda x: x['id'])
        
        # 3. Run Test Data
        test_results = []
        correct = 0
        total = len(test_emails)
        for email, subject, expected_spam in test_emails:
            prediction, cluster_id, confidence = detector.predict(email, subject)
            expected_label = 'spam' if expected_spam else 'ham'
            is_correct = (prediction == expected_label)
            if is_correct: correct += 1
            test_results.append({
                'email': email[:50] + "...",
                'subject': subject,
                'prediction': prediction,
                'expected': expected_label,
                'cluster': cluster_id,
                'correct': is_correct
            })
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # 4. Graph Generation
        graph_data = {'nodes_vectors': [], 'nodes_clusters': [], 'edges': [], 'height': 0}
        cluster_node_map = {}
        y_step_c = 45
        current_y_c = 40
        x_c = 600 # X-position for clusters (right side)
        
        for clust in clusters_list:
            node_c = {
                'id': f"c_{clust['id']}",
                'x': x_c,
                'y': current_y_c,
                'name': f"Cluster {clust['id']}" if clust['id'] != -1 else "Unclustered",
                'type': clust['label']
            }
            graph_data['nodes_clusters'].append(node_c)
            cluster_node_map[clust['id']] = node_c
            current_y_c += y_step_c

        y_step_v = 30
        current_y_v = 30
        x_v = 200 # X-position for vectors (left side)
        
        for i, vec in enumerate(vector_assignments):
            # --- UPDATED: Pass subject and body to the node data ---
            node_v = {
                'id': vec['id'],
                'x': x_v,
                'y': current_y_v,
                'name': vec['name'],
                'type': vec['type'],
                'subject': vec['subject'],
                'body': vec['body']
            }
            graph_data['nodes_vectors'].append(node_v)
            
            end_node = cluster_node_map.get(vec['cluster_id'])
            if end_node:
                edge = {
                    'x1': node_v['x'], 'y1': node_v['y'],
                    'x2': end_node['x'], 'y2': end_node['y'],
                    'type': vec['type']
                }
                graph_data['edges'].append(edge)
            
            current_y_v += y_step_v
        
        graph_data['height'] = max(current_y_v, current_y_c) + 20 
        
        # 5. Package all data
        results_data = {
            'stats': stats,
            'clusters': clusters_list,
            'test_results': test_results,
            'accuracy': f"{accuracy:.1f}",
            'correct': correct,
            'total': total
        }

    return render_template(
        'index.html', 
        vigilance_default=vigilance_default, 
        results=results_data,
        explanation_data=explanation_data,
        graph_data=graph_data
    )

if __name__ == '__main__':
    app.run(debug=True)