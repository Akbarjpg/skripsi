"""
Face Recognition CNN Model for Step 4
Implements CNN-based face recognition using transfer learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
import os
from pathlib import Path
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logger import get_logger


class FaceRecognitionCNN(nn.Module):
    """
    CNN model for face recognition using transfer learning
    Based on ResNet50 architecture with custom embedding layer
    """
    
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        """
        Initialize face recognition CNN
        
        Args:
            embedding_dim: Dimension of face embeddings (default: 128)
            pretrained: Use pretrained ResNet50 weights
        """
        super(FaceRecognitionCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.logger = get_logger(__name__)
        
        # Load pretrained ResNet50
        self.backbone = resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add custom embedding layers
        self.embedding_layers = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize custom layer weights"""
        for module in self.embedding_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, 224, 224)
            
        Returns:
            Face embeddings (batch_size, embedding_dim)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Generate embeddings
        embeddings = self.embedding_layers(features)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class FaceRecognitionSystem:
    """
    Complete face recognition system with database integration
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 embedding_dim: int = 128,
                 similarity_threshold: float = 0.85):
        """
        Initialize face recognition system
        
        Args:
            model_path: Path to saved model weights
            embedding_dim: Dimension of face embeddings
            similarity_threshold: Minimum similarity for recognition
        """
        self.logger = get_logger(__name__)
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = FaceRecognitionCNN(embedding_dim=embedding_dim)
        self.model.to(self.device)
        
        # Load model weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.logger.warning("No model weights loaded - using random initialization")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # In-memory embedding cache
        self.embedding_cache = {}
        self.user_cache = {}
        
    def load_model(self, model_path: str) -> bool:
        """
        Load model weights from file
        
        Args:
            model_path: Path to model weights
            
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """
        Save model weights to file
        
        Args:
            model_path: Path to save model weights
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'embedding_dim': self.embedding_dim,
                'similarity_threshold': self.similarity_threshold
            }
            
            torch.save(checkpoint, model_path)
            self.logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for model input
        
        Args:
            face_image: Face image as numpy array (BGR)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        if len(face_image.shape) == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(face_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from image
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Face embedding or None if extraction fails
        """
        try:
            # Preprocess image
            tensor = self.preprocess_face(face_image)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to extract embedding: {e}")
            return None
    
    def calculate_similarity(self, embedding1: np.ndarray, 
                           embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Reshape for sklearn
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def register_face(self, face_images: List[np.ndarray], 
                     user_id: str) -> Dict[str, any]:
        """
        Register a new user with multiple face images
        
        Args:
            face_images: List of face images
            user_id: Unique user identifier
            
        Returns:
            Registration result dictionary
        """
        try:
            if len(face_images) < 5:
                return {
                    'success': False,
                    'message': 'Need at least 5 face images for registration'
                }
            
            # Extract embeddings from all images
            embeddings = []
            for face_image in face_images:
                embedding = self.extract_embedding(face_image)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if len(embeddings) < 3:
                return {
                    'success': False,
                    'message': 'Could not extract enough valid embeddings'
                }
            
            # Calculate average embedding
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Normalize the average embedding
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            # Store in cache
            self.embedding_cache[user_id] = avg_embedding
            self.user_cache[user_id] = {
                'embedding': avg_embedding,
                'registration_time': time.time(),
                'num_images': len(embeddings)
            }
            
            self.logger.info(f"User {user_id} registered with {len(embeddings)} embeddings")
            
            return {
                'success': True,
                'message': f'User registered successfully with {len(embeddings)} images',
                'embedding': avg_embedding,
                'num_embeddings': len(embeddings)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to register user: {e}")
            return {
                'success': False,
                'message': f'Registration failed: {str(e)}'
            }
    
    def recognize_face(self, face_image: np.ndarray) -> Dict[str, any]:
        """
        Recognize face from image
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Recognition result dictionary
        """
        try:
            start_time = time.time()
            
            # Extract embedding from input image
            query_embedding = self.extract_embedding(face_image)
            if query_embedding is None:
                return {
                    'success': False,
                    'user_id': 'Unknown',
                    'confidence': 0.0,
                    'message': 'Could not extract face embedding'
                }
            
            if not self.embedding_cache:
                return {
                    'success': False,
                    'user_id': 'Unknown',
                    'confidence': 0.0,
                    'message': 'No registered users found'
                }
            
            # Compare with all registered embeddings
            best_match = None
            best_similarity = 0.0
            
            for user_id, stored_embedding in self.embedding_cache.items():
                similarity = self.calculate_similarity(query_embedding, stored_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = user_id
            
            processing_time = time.time() - start_time
            
            # Check if similarity meets threshold
            if best_similarity >= self.similarity_threshold:
                self.logger.info(f"Face recognized: {best_match} ({best_similarity:.3f})")
                return {
                    'success': True,
                    'user_id': best_match,
                    'confidence': float(best_similarity),
                    'message': 'Face recognized successfully',
                    'processing_time': processing_time
                }
            else:
                self.logger.info(f"Face not recognized (best: {best_similarity:.3f})")
                return {
                    'success': False,
                    'user_id': 'Unknown',
                    'confidence': float(best_similarity),
                    'message': f'No match found (best similarity: {best_similarity:.3f})',
                    'processing_time': processing_time
                }
                
        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")
            return {
                'success': False,
                'user_id': 'Unknown',
                'confidence': 0.0,
                'message': f'Recognition failed: {str(e)}'
            }
    
    def load_embeddings_from_database(self, database_embeddings: Dict[str, np.ndarray]):
        """
        Load user embeddings from database into cache
        
        Args:
            database_embeddings: Dictionary of user_id -> embedding
        """
        try:
            self.embedding_cache.clear()
            self.user_cache.clear()
            
            for user_id, embedding in database_embeddings.items():
                self.embedding_cache[user_id] = embedding
                self.user_cache[user_id] = {
                    'embedding': embedding,
                    'loaded_from_db': True,
                    'load_time': time.time()
                }
            
            self.logger.info(f"Loaded {len(database_embeddings)} embeddings from database")
            
        except Exception as e:
            self.logger.error(f"Failed to load embeddings from database: {e}")
    
    def get_system_info(self) -> Dict[str, any]:
        """
        Get system information and statistics
        
        Returns:
            System information dictionary
        """
        return {
            'model_info': {
                'embedding_dim': self.embedding_dim,
                'similarity_threshold': self.similarity_threshold,
                'device': str(self.device),
                'model_parameters': sum(p.numel() for p in self.model.parameters())
            },
            'cache_info': {
                'registered_users': len(self.embedding_cache),
                'user_ids': list(self.embedding_cache.keys())
            },
            'system_status': {
                'model_loaded': True,
                'cache_loaded': len(self.embedding_cache) > 0,
                'ready_for_recognition': len(self.embedding_cache) > 0
            }
        }


class FaceRecognitionTrainer:
    """
    Training utilities for face recognition model
    """
    
    def __init__(self, model: FaceRecognitionCNN, device: torch.device):
        """
        Initialize trainer
        
        Args:
            model: Face recognition model
            device: Training device
        """
        self.model = model
        self.device = device
        self.logger = get_logger(__name__)
    
    def triplet_loss(self, anchor: torch.Tensor, positive: torch.Tensor, 
                    negative: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
        """
        Compute triplet loss for face recognition
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same person)
            negative: Negative embeddings (different person)
            margin: Margin for triplet loss
            
        Returns:
            Triplet loss
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + margin)
        return loss.mean()
    
    def fine_tune_model(self, train_loader, val_loader, num_epochs: int = 10):
        """
        Fine-tune the model on face recognition data
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
        """
        # This is a placeholder for actual training implementation
        # In a real scenario, you would implement the full training loop
        self.logger.info("Fine-tuning implementation would go here")
        pass
