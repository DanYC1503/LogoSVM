all:
	export LD_LIBRARY_PATH=/home/danyanc/opencv/opencvi/lib:/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH && \
	g++ Principal.cpp functions.cpp training.cpp -std=c++17 \
	-I/home/danyanc/opencv/opencvi/include/opencv4/ \
	-I/usr/local/cuda/include \
	-L/home/danyanc/opencv/opencvi/lib \
	-L/usr/local/cuda/lib64 \
	-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
	-lopencv_video -lopencv_videoio -lopencv_features2d -lopencv_objdetect \
	-lopencv_ml -lopencv_dnn \
	-ltinyxml2 \
	-lcuda -lcudart \
	-o vision.bin


run:
	export LD_LIBRARY_PATH=/home/danyanc/opencv/opencvi/lib:$$LD_LIBRARY_PATH && \
	./vision.bin
