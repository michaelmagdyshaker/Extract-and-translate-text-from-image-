# Extract-and-translate-text-from-image

This project addresses the problem of the lack of knowledge between people of English language which is nowadays used in almost everything in travelling abroad, in many products catalogs, in books, supermarkets and many more, also it addresses the problem of wasting time in trying to translate a sentence on keyboard.

And it solves that problem by being able to translate the text in any image. This is a very difficult task since there are always problems either due to the way of extracting the text from the image or the technique it is recognized with. We present a fully automatic approach that uses deep neural networks to extract any text in any image and get it translate from English to Arabic.

We explore advanced convolutional neural network methods, our used models are VGG: Visual Geometry Group as feature extractor and SSD: Single Shot Multi Box Detector models and we trained them using supervised learning with Batch Mode. We also explored loss function (cross entropy) and activation functions (ReLU and Softmax) to understand the best practices to obtain the most pleasing detection and recognition output of the text in the image. 

In particular, we find that our model that uses cross entropy as a loss, ReLU and softmax activation function and VGG and SSD for text detection and for character recognition seems to perform the desired target in an efficient way. 
 
Furthermore, our architecture can process real nature images of different resolution and yet be able to detect the text on them. We demonstrate our method extensively on many different text styles and fonts, including different backgrounds and different colors and it still be able to recognize the characters and translate the full word. We also worked on the feature of capturing images with mobile not only scanned images. Finally we offered the user the luxury of a user-friendly interface. 

Finally, we get an efficient results in detecting every text in different location in the image then recognizing every character and finally get the word translated with the luxury of displaying the translated word back on the image.   


