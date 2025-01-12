import cv2
import numpy as np
import mediapipe as mp
import torch
import pyautogui
import time
from torch import nn
from Label_Dict import label_dict_from_config_file

class HandLandmarksDetector():
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(False,max_num_hands=1,min_detection_confidence=0.5)

    def detectHand(self,frame):
        hands = []
        frame = cv2.flip(frame, 1)
        annotated_image = frame.copy()
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                for landmark in hand_landmarks.landmark:
                    x,y,z = landmark.x,landmark.y,landmark.z
                    hand.extend([x,y,z])
            hands.append(hand)
        return hands,annotated_image


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        list_label = label_dict_from_config_file("hand_gesture.yaml")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, len(list_label)),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    def predict(self,x,threshold=0.9):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        chosen_ind = torch.argmax(softmax_prob,dim=1)
        return torch.where(softmax_prob[0,chosen_ind]>threshold,chosen_ind,-1)
    def predict_with_known_class(self,x):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        return torch.argmax(softmax_prob,dim=1)
    def score(self,logits):
        return -torch.amax(logits,dim=1)

class LightGesture:
    def __init__(self,model_path):
        self.height = 720
        self.width = 1280

        self.detector = HandLandmarksDetector()
        self.status_text = None
        self.signs = label_dict_from_config_file("hand_gesture.yaml")
        self.classifier = NeuralNetwork()
        self.classifier.load_state_dict(torch.load(model_path))
        self.classifier.eval()

        self.prediction_buffer = []
        self.buffer_size = 12

        self.turn_off = False
        self.scroll_down = False
        self.scroll_up = False
        self.like = False
        self.reset = False
    
    def check_stable_gesture(self, prediction):
        self.prediction_buffer.append(prediction)
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
        if self.prediction_buffer.count(prediction) > (self.buffer_size*0.8):
            return True
        return False

    def run(self):
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Could not open camera.")
            return  
        cam.set(3,1280)
        cam.set(4,720)
        while cam.isOpened():
            _,frame = cam.read()
            hand,img = self.detector.detectHand(frame)
            if len(hand) != 0:
                with torch.no_grad():
                    hand_landmark = torch.from_numpy(np.array(hand[0],dtype=np.float32).flatten()).unsqueeze(0)
                    class_number = self.classifier.predict(hand_landmark).item()
                    if class_number != -1:
                        self.status_text = self.signs[class_number]
                        if self.check_stable_gesture(self.status_text):
                            if self.status_text == "turn_off" and not self.turn_off:
                                pyautogui.hotkey('ctrl','w')
                                self.turn_off = True
                            elif self.status_text == "scroll_down" and not self.scroll_down:
                                pyautogui.press('down')
                                self.scroll_down = True
                                time.sleep(0.5)
                                self.scroll_down = False
                            elif self.status_text == "scroll_up" and not self.scroll_up:
                                pyautogui.press('up')
                                self.scroll_up = True
                                time.sleep(0.5)
                                self.scroll_up = False
                            elif self.status_text == "like" and not self.like:
                                pyautogui.press("L") 
                                self.like = True
                                time.sleep(0.5)
                                self.like = False
                            elif self.status_text == "reset" and not self.reset:
                                pyautogui.press("f5")
                                self.reset = True
                                time.sleep(0.5)
                                self.reset = False
            cv2.imshow('Camera',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = ".\\models\\model_10-01 17_47_NeuralNetWork_best"
    light = LightGesture(model_path)
    light.run()