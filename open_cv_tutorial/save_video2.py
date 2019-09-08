import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20, (640, 480))

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    # get frame size
    print('Frame_width: ', frame.shape[1])
    print('Frame_height: ', frame.shape[0])

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # write the flipped frame
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('frame', gray)

    # quit() by enter "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
