rm(list=ls())
setwd("C:/Users/pratik/Desktop/Face detection")
train_y<-read.csv("label.csv")
test_y<-read.csv("TestLabel.csv")
require(EBImage)  
setwd("C:/Users/pratik/Desktop/Face detection/Resized1")
save_in <- "C:/Users/pratik/Desktop/Face detection/Resized1"
images <- list.files() #Load Images names 
w<-200
h<-200
# Main loop resize images and set them to greyscale
for(i in 1:length(images))
{
  result <- tryCatch({
    imgname <- images[i]#Storing Image Name into Imgname
    img <- readImage(imgname)#reading the Images
    img_resized <- resize(img, w = w, h = h)#resezing the images
    grayimg <- channel(img_resized,"gray")
    path <- paste(save_in, imgname, sep = "")#providing the path to save the images
    writeImage(grayimg, path, quality = 70)#saving the edited images
    print(paste("Done",i,sep = " "))},
    error = function(e){print(e)})
  
}
setwd("C:/Users/pratik/Desktop/Face detection/Resized1")#Resized files path
out_file <-"images.csv"
images <- list.files()
df <- data.frame()
img_size <- 200*200
for(i in 1:length(images))
{
  # Read image
  img <- readImage(images[i])
  # Get the image as a matrix
  img_matrix <-as.matrix(img@.Data) 
  # Coerce to a vector
  img_vector <- as.vector(t(img_matrix))
  # Add label
  vec <- c( img_vector)
  # Bind rows
  df <- rbind(df,vec)
  # Print status info
  print(paste("Done ", i, sep = ""))
}
names(df) <- c(paste("pixel", c(1:img_size))) # Set names
##write.csv(df, out_file, row.names = FALSE)
df<-cbind(train_y,df)
train<-df
train <- data.matrix(train)
train_x <- t(train[, -1])
train_y <- train[, 1]
train_array <- train_x
dim(train_array) <- c(200, 200, 1, ncol(train_x))

#Validation images 
setwd("C:/Users/pratik/Desktop/Face detection/F2")
images1 <- list.files()
df1 <- data.frame()
img_size <- 200*200
for(i in 1:length(images1))
{
  # Read image
  img1 <- readImage(images1[i])
  # Get the image as a matrix
  img_matrix1 <- img1@.Data
  # Coerce to a vector
  img_vector1 <- as.vector(t(img_matrix1))
  # Add label
  vec <- c(img_vector1)
  # Bind rows
  df1 <- rbind(df1,vec)
  # Print status info
  print(paste("Done ", i, sep = ""))
}
names(df1) <- c(paste("pixel", c(1:img_size)))
df1<-cbind(test_y,df1)
test<-df1
test_x <- t(test[, -1])
test_y <- test[, 1]
test_array <- test_x
dim(test_array) <- c(200, 200, 1, ncol(test_x))

# Set up the symbolic model
#-------------------------------------------------------------------------------
# creating the variable data in mxnet format
data <- mx.symbol.Variable('data') #This is how we initialize the architecture

# kernel represnets the size of kernel and num_filter represents the number of kernels
# act_type is the activation function
# stride is the size with which the kernel shape (to shift on data matrix)

# 1st convolutional layer  5x5 kernel and 16 filters
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 16)
relu_1 <- mx.symbol.Activation(data = conv_1, act_type = "relu")
pool_1 <- mx.symbol.Pooling(data = relu_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

# 2nd convolutional layer 5x5 kernel and 32 filters
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 32)
relu_2 <- mx.symbol.Activation(data = conv_2, act_type = "relu")
pool_2 <- mx.symbol.Pooling(data=relu_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

# 1st fully connected layer (Input & Hidden layers)
flatten <- mx.symbol.Flatten(data = pool_2)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 128)
relu_3 <- mx.symbol.Activation(data = fc_1, act_type = "relu")


# 2nd fully connected layer(Hidden & Output layers)
fc_2 <- mx.symbol.FullyConnected(data = relu_3, num_hidden = 10)
# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)

# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
mx.set.seed(100)

# Create a mxnet CPU context. - Device used. CPU in this case, but not the GPU 
devices <- mx.cpu()

# Training
#-------------------------------------------------------------------------------

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     eval.data = list(data = test_array, label = test_y),
                                     ctx = devices,
                                     num.round = 6, # number of epochs
                                     array.batch.size = 60,
                                     learning.rate = 0.01,
                                     momentum = 0.5,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

