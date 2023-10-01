import openai
import speech_recognition as sr
import pyttsx3
import cv2

api_key = ''

thres = 0.45
def detect_objects_and_generate_prompt():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    cap.set(3, 400)
    cap.set(4, 720)
    cap.set(10, 70)

    classNames = []
    classFile = 'coco.names'  # Make sure you have a file named coco.names with class names
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(100, 100)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    detected_items = []
    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
    
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                detected_item = classNames[classId - 1].upper()
                detected_items.append(detected_item)  # Append detected item to the list
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, detected_item, (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                if detected_items:
                   detected_items_string = ", ".join(detected_items) 
                   print(detected_items)
                   return detected_items_string
            cv2.imshow("Output", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

    cap.release()
    cv2.destroyAllWindows()


def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for your prompt...")
        recognizer.adjust_for_ambient_noise(source)  
        audio = recognizer.listen(source, timeout=5)  

    try:
        prompt = recognizer.recognize_google(audio)  
        return prompt
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


def generate_and_play_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def get_input():
    print("How would you like to provide your input?")
    print("1. Voice")
    print("2. Direct Text")
    print("3. from camera and voice")
    choice = input("Enter your choice : ")

    if choice == '1':
        return recognize_speech()
    elif choice == '2':
        return input("Enter your text input: ")
    elif choice == '3':
        return recognize_speech() + " " + detect_objects_and_generate_prompt()
    else:
        print("Invalid choice. Please enter '1' for voice or '2' for direct text.")
        return get_input()

# Get the spoken prompt or direct text input from the user
user_input = get_input()

# Make a request to the GPT-3 API using the input
if user_input:
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can choose a different engine based on your needs
        prompt=user_input,
        max_tokens=50,  # Adjust the number of tokens based on your desired response length
        api_key=api_key
    )

    # Extract and print the generated response
    generated_response = response.choices[0].text
    print(f"Generated Response: {generated_response}")

    # Generate and play speech from the generated response
    generate_and_play_speech(generated_response)
else:
    print("No valid input received.")
