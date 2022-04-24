import cv2

import torch
from model import FireNet
from torchvision import transforms

cap = cv2.VideoCapture("./videos/fire_flames_heat_red_hot_bonfire_1062.mp4")
#cap = cv2.VideoCapture(0)

net = FireNet()
net.float()
net.load_state_dict(torch.load('./trained_weights100.pth', map_location=torch.device('cpu')))
net.eval()

while cap.isOpened():

    # Process frame
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))

    # Input frame into network
    with torch.no_grad():
        transform = transforms.ToTensor()
        image = transform(image)
        image = image.unsqueeze(0)
        output = net(image.float())
        predicted = torch.nn.functional.softmax(output)
        prob = predicted[0][0].item()*100.0
        cv2.putText(frame, "Fire prob: %.3f" % prob, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('real time', frame)

    if cv2.waitKey(25) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()