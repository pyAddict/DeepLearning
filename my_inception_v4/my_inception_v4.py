from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K

if K.image_dim_ordering() == 'th':
		channel_axis = 1
else:
	channel_axis = -1

#subsample => strides
#channel_axis => batch normalize axis
def conv(x,nb_filter, nb_row, nb_col, border_mode='same',subsample=(1,1), bias=False):
	x = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample,border_mode=border_mode,bias=bias)(x)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)
	return x


#takes input of 149x149x3 or 3x149x149
def stem(ip):
	x1 = conv(x=ip,nb_filter=32,nb_row=3,nb_col=3,border_mode='same',subsample=(1,1))
	x1 = conv(x=x1,nb_filter=32,nb_row=3,nb_col=3,border_mode='valid',subsample=(1,1))
	x1 = conv(x=x1,nb_filter=64,nb_row=3,nb_col=3,border_mode='same',subsample=(1,1))
	x11 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x1)
	x12 = conv(x=x1,nb_filter=96,nb_row=3,nb_col=3,border_mode='valid',subsample=(2,2))
	s1 = merge([x11, x12], mode='concat', concat_axis=channel_axis)

	x2 = conv(x=s1,nb_filter=64,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))
	x2 = conv(x=x2,nb_filter=96,nb_row=3,nb_col=3,border_mode='valid',subsample=(1,1))

	x3 = conv(x=s1,nb_filter=64,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))
	x3 = conv(x=x3,nb_filter=64,nb_row=7,nb_col=1,border_mode='same',subsample=(1,1))
	x3 = conv(x=x3,nb_filter=64,nb_row=1,nb_col=7,border_mode='same',subsample=(1,1))
	x3 = conv(x=x3,nb_filter=96,nb_row=3,nb_col=3,border_mode='valid',subsample=(1,1))

	s2 = merge([x2,x3], mode='concat', concat_axis=channel_axis)

	x4 = conv(x=s2,nb_filter=192,nb_row=3,nb_col=3,border_mode='valid',subsample=(2,2))
	x5 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(s2)
	s3 = merge([x4,x5], mode='concat', concat_axis=channel_axis)

	return s3


#Handles 35x35 size input
def INCEPTION_A(ip):
	x1 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(ip)
	x1 = conv(x=x1,nb_filter=96,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))

	x2 = conv(x=ip,nb_filter=96,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))

	x3 = conv(x=ip,nb_filter=64,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))
	x3 = conv(x=x3,nb_filter=96,nb_row=3,nb_col=3,border_mode='same',subsample=(1,1))

	x4 = conv(x=ip,nb_filter=64,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))
	x4 = conv(x=x4,nb_filter=96,nb_row=3,nb_col=3,border_mode='same',subsample=(1,1))
	x4 = conv(x=x4,nb_filter=96,nb_row=3,nb_col=3,border_mode='same',subsample=(1,1))

	i_A =  merge([x1,x2,x3,x4], mode='concat', concat_axis=channel_axis)

	return i_A


#Handles 17x17 size input
def INCEPTION_B(ip):
	x1 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(ip)
	x1 = conv(x=x1,nb_filter=128,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))

	x2 = conv(x=ip,nb_filter=384,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))

	x3 = conv(x=ip,nb_filter=192,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))
	x3 = conv(x=x3,nb_filter=224,nb_row=1,nb_col=7,border_mode='same',subsample=(1,1))
	x3 = conv(x=x3,nb_filter=256,nb_row=7,nb_col=1,border_mode='same',subsample=(1,1))

	x4 = conv(x=ip,nb_filter=192,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))
	x4 = conv(x=x4,nb_filter=192,nb_row=1,nb_col=7,border_mode='same',subsample=(1,1))
	x4 = conv(x=x4,nb_filter=224,nb_row=7,nb_col=1,border_mode='same',subsample=(1,1))
	x4 = conv(x=x4,nb_filter=224,nb_row=1,nb_col=7,border_mode='same',subsample=(1,1))
	x4 = conv(x=x4,nb_filter=256,nb_row=7,nb_col=1,border_mode='same',subsample=(1,1))

	i_B = merge([x1,x2,x3,x4], mode='concat', concat_axis=channel_axis)

	return i_B


#Handles 8x8 size input
def INCEPTION_C(ip):
	x1 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(ip)
	x1 = conv(x=x1,nb_filter=256,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))

	x2 = conv(x=ip,nb_filter=256,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))

	x3 = conv(x=x1,nb_filter=384,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))
	x31 = conv(x=x3,nb_filter=256,nb_row=1,nb_col=3,border_mode='same',subsample=(1,1))
	x32 = conv(x=x3,nb_filter=256,nb_row=3,nb_col=1,border_mode='same',subsample=(1,1))

	x4 = conv(x=x1,nb_filter=384,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))
	x4 = conv(x=x4,nb_filter=448,nb_row=1,nb_col=3,border_mode='same',subsample=(1,1))
	x4 = conv(x=x4,nb_filter=512,nb_row=3,nb_col=1,border_mode='same',subsample=(1,1))
	x41 = conv(x=x4,nb_filter=256,nb_row=3,nb_col=1,border_mode='same',subsample=(1,1))
	x42 = conv(x=x4,nb_filter=256,nb_row=1,nb_col=3,border_mode='same',subsample=(1,1))

	i_C = merge([x1,x2,x31,x32,x41,x42], mode='concat', concat_axis=channel_axis)

	return i_C

#Used to reduce from 35x35 to 17x17
def REDUCION_A(ip):
	#For inception_v4 dimention of reduction_A are
	k = 192
	l = 224
	m = 256
	n = 384
	x1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(ip)


	x2 = conv(x=ip,nb_filter=n,nb_row=3,nb_col=3,border_mode='valid',subsample=(2,2))

	x3 = conv(x=ip,nb_filter=k,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))
	x3 = conv(x=x3,nb_filter=l,nb_row=3,nb_col=3,border_mode='same',subsample=(1,1))
	x3 = conv(x=x3,nb_filter=m,nb_row=3,nb_col=3,border_mode='valid',subsample=(2,2))
	r_A = merge([x1,x2,x3], mode='concat', concat_axis=channel_axis)

	return r_A	

def REDUCION_B(ip):
	x1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(ip)

	x2 = conv(x=ip,nb_filter=192,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))

	x2 = conv(x=x2,nb_filter=192,nb_row=3,nb_col=3,border_mode='valid',subsample=(2,2))

	x3 = conv(x=ip,nb_filter=256,nb_row=1,nb_col=1,border_mode='same',subsample=(1,1))
	x3 = conv(x=x3,nb_filter=256,nb_row=1,nb_col=7,border_mode='same',subsample=(1,1))
	x3 = conv(x=x3,nb_filter=320,nb_row=7,nb_col=1,border_mode='same',subsample=(1,1))
	x3 = conv(x=x3,nb_filter=320,nb_row=3,nb_col=3,border_mode='valid',subsample=(2,2))
	r_B = merge([x1,x2,x3], mode='concat', concat_axis=channel_axis)

	return r_B



def create_model_Inception_v4():
	nb_classes = 5
	nb_channels = 3
	nb_height = 149
	nb_width = 149
	if K.image_dim_ordering() == 'th':
		dim = Input((nb_channels, nb_height, nb_width))
	else:
		dim = Input((nb_height, nb_width, nb_channels))

	x = stem(dim)

	for fourtimes in range(4):
		x = INCEPTION_A(x)

	x = REDUCION_A(x)

	for seventimes in range(7):
		x = INCEPTION_B(x)

	x = REDUCION_B(x)

	for threetimes in range(3):
		x = INCEPTION_C(x)

	x = AveragePooling2D((8,8))(x)

	x = Dropout(0.8)(x)
	x =Flatten()(x)

	output = Dense(output_dim=nb_classes, activation='softmax')(x)

	model = Model(dim, output, name='Inception-v4')

	return model



# if __name__ == '__main__':
# 	model = create_model_Inception_v4()
# 	model.summary()
# 	#plot(inception_v4, to_file="Inception-v4.png", show_shapes=True)





























