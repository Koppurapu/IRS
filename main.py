
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler

class ImageRetrievalSystem:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = []
        self.features = []
        self.labels = []
        self.scaler = StandardScaler()
        self.model = svm.SVC(kernel='linear', probability=True)

    def load_images(self):
        print("Loading images...")
        for file in os.listdir(self.image_folder):
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                path = os.path.join(self.image_folder, file)
                img = cv2.imread(path)
                if img is not None:
                    self.images.append((file, img))
                    feat = self.extract_features(img)
                    self.features.append(feat)
        self.features = np.array(self.features)
        self.features = self.scaler.fit_transform(self.features)
        print("Loaded", len(self.images), "images")

    def extract_features(self, img):
        hist = cv2.calcHist([img],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
        return hist.flatten()

    def train(self):
        if len(self.labels) < 2:
            print("Need more feedback")
            return
        self.model.fit(self.features, self.labels)

    def retrieve(self):
        probs = self.model.predict_proba(self.features)[:,1]
        ranked = sorted(zip(self.images, probs), key=lambda x: x[1], reverse=True)
        return ranked

    def feedback(self, feedback_dict):
        self.labels = []
        for name, img in self.images:
            if name in feedback_dict:
                self.labels.append(feedback_dict[name])
            else:
                self.labels.append(0)

def main():
    folder = "dataset"
    ir = ImageRetrievalSystem(folder)
    ir.load_images()

    feedback = {}
    print("\nRound 1: Provide feedback (1 relevant, -1 irrelevant, 0 skip)")
    for name, img in ir.images:
        cv2.imshow(name, img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == ord('1'):
            feedback[name] = 1
        elif key == ord('-'):
            feedback[name] = -1
        else:
            feedback[name] = 0

    ir.feedback(feedback)
    ir.train()

    results = ir.retrieve()
    print("\nRanked Results:")
    for (name, img), score in results:
        print(name, "Score:", score)

if __name__ == "__main__":
    main()
