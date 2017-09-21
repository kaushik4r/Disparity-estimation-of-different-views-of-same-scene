import cv2
import numpy as np
import time

start_time = time.time()

img_1 = cv2.imread('view1.png',0)
img_5 = cv2.imread('view5.png',0)

orig_rows, orig_cols = img_1.shape


def three_cross_three_block():

	padded_image_1 = cv2.copyMakeBorder(img_1,1,1,1,1, cv2.BORDER_CONSTANT, value = 0)
	padded_image_5 = cv2.copyMakeBorder(img_5,1,1,1,1, cv2.BORDER_CONSTANT, value = 0)

	ground_truth_1 = cv2.imread('disp1.png',0)
	ground_truth_5 = cv2.imread('disp5.png',0)
	
	rows_img_1,cols_img_1 = padded_image_1.shape
	rows_img_5,cols_img_5 = padded_image_5.shape

	disp_array_1 = np.zeros((rows_img_1,cols_img_1))
	disp_array_5 = np.zeros((rows_img_5,cols_img_5))
	
	# Getting the disparity matrix for view1
	for i in range(1, rows_img_1-1):
		for j in range(1,cols_img_1-1):

			sub_matrix_1 = padded_image_1[i-1:i+2,j-1:j+2]
			min_arr = []
			for k in range(j-75,j):
				if (k<1):
					k=1
				sub_matrix_5 = padded_image_5[i-1:i+2,k-1:k+2]

				subtract_matrix = np.subtract(sub_matrix_1, sub_matrix_5)
				SSD_value = np.sum(np.square(subtract_matrix))
				min_arr.append([SSD_value,k])

			min_value = min(min_arr)
			disp = j - (min_value[1])
			disp_array_1[i][j] = disp


	disp_img1 = disp_array_1/disp_array_1.max()
	cv2.imshow('Disparity map 1', disp_img1)
	cv2.waitKey(0)

	disp_array_1_mse = disp_array_1[1:rows_img_1-1, 1:cols_img_1-1]

	mse_disp1 = (np.sum((ground_truth_1-disp_array_1_mse)**2))/(rows_img_1*cols_img_1)
	print "mse value for view1 for 3*3 is ",mse_disp1
	
	# Getting the disparity matrix for view5
	for i in range(1, rows_img_5-1):
		for j in range(1,cols_img_5-1):

			sub_matrix_5 = padded_image_5[i-1:i+2,j-1:j+2]
			min_arr = []
			for k in range(j,j+75):
				if (k >= cols_img_5-1 ):
					k=cols_img_5-2
				sub_matrix_1 = padded_image_1[i-1:i+2,k-1:k+2]

				subtract_matrix = np.subtract(sub_matrix_1, sub_matrix_5)
				SSD_value = np.sum(np.square(subtract_matrix))
				min_arr.append([SSD_value,k])

			min_value = min(min_arr)
			disp = abs(j - (min_value[1]))
			disp_array_5[i][j] = disp


	disp_img5 = disp_array_5/disp_array_5.max()
	cv2.imshow('Disparity map 2', disp_img5)
	cv2.waitKey(0)

	disp_array_5_mse = disp_array_5[1:rows_img_5-1, 1:cols_img_5-1]
	mse_disp5 = (np.sum((ground_truth_5-disp_array_5_mse)**2))/(rows_img_5*cols_img_5)
	print "mse value for view5 for 3*3 is ",mse_disp5
	
	# Consistency check
	disp_array_consistency_1 = np.zeros((rows_img_1,cols_img_1))
	disp_array_consistency_5 = np.zeros((rows_img_1,cols_img_1))

	
	for i in range(1,rows_img_5-1):
		for j in range(1,cols_img_5-1):

			if (abs(j - int(disp_array_1[i][j])) >= 0):
				if (disp_array_1[i][j] == disp_array_5[i][abs(j - int(disp_array_1[i][j]))]):
					disp_array_consistency_1[i][j] = disp_array_1[i][j]
				else:
					disp_array_consistency_1[i][j] = 0

			if (abs(j - int(disp_array_5[i][j])) >= 0):
				if (disp_array_5[i][j] == disp_array_1[i][abs(j + int(disp_array_5[i][j]))]):
					disp_array_consistency_5[i][j] = disp_array_5[i][j]
				else:
					disp_array_consistency_5[i][j] = 0


	disp_img_consistency_1 = disp_array_consistency_1/disp_array_consistency_1.max()
	disp_img_consistency_5 = disp_array_consistency_5/disp_array_consistency_5.max()
	cv2.imshow('Disparity map for left image after consistency check for 3*3 block', disp_img_consistency_1)
	cv2.waitKey(0)
	cv2.imshow('Disparity map for right image after consistency check for 3*3 block', disp_img_consistency_5)
	cv2.waitKey(0)

	consistency_arr_1 = []
	consistency_arr_5 = []

	for i in range(1,rows_img_5-1):
		for j in range(1,cols_img_5-1):
			if (int(disp_array_consistency_1[i][j]) != 0) and ((int(disp_array_consistency_5[i][j]) != 0)) and (i < 370 and j < 463):
				
				consistency_arr_1.append(ground_truth_1[i][j]-disp_array_consistency_1[i][j])
				consistency_arr_5.append(ground_truth_5[i][j]-disp_array_consistency_5[i][j])

	consistency_arr_1 = np.array(consistency_arr_1)
	consistency_arr_5 = np.array(consistency_arr_5)

	mse_disp1_consistency = (np.sum(consistency_arr_1**2))/(rows_img_1*cols_img_1)
	mse_disp5_consistency = (np.sum(consistency_arr_5**2))/(rows_img_5*cols_img_5)

	print "mse_disp1_consistency =", mse_disp1_consistency
	print "mse_disp5_consistency =", mse_disp5_consistency

def nine_cross_nine_block():

	padded_image_1 = cv2.copyMakeBorder(img_1,4,4,4,4, cv2.BORDER_CONSTANT, value = 0)
	padded_image_5 = cv2.copyMakeBorder(img_5,4,4,4,4, cv2.BORDER_CONSTANT, value = 0)

	ground_truth_1 = cv2.imread('disp1.png',0)
	ground_truth_5 = cv2.imread('disp5.png',0)

	rows_img_1,cols_img_1 = padded_image_1.shape
	rows_img_5,cols_img_5 = padded_image_5.shape

	disp_array_1 = np.zeros((rows_img_1,cols_img_1))
	disp_array_5 = np.zeros((rows_img_5,cols_img_5))
	
	# Getting the disparity matrix for view1
	for i in range(4, rows_img_1-4):
		for j in range(4,cols_img_1-4):

			sub_matrix_1 = padded_image_1[i-4:i+5,j-4:j+5]
			min_arr = []
			for k in range(j-75,j):
				if (k<4):
					k=4
				sub_matrix_5 = padded_image_5[i-4:i+5,k-4:k+5]

				subtract_matrix = np.subtract(sub_matrix_1, sub_matrix_5)
				SSD_value = np.sum(np.square(subtract_matrix))
				min_arr.append([SSD_value,k])

			min_value = min(min_arr)
			disp = j - (min_value[1])
			disp_array_1[i][j] = disp


	disp_img1 = disp_array_1/disp_array_1.max()
	cv2.imshow('Disparity map 1', disp_img1)
	cv2.waitKey(0)

	disp_array_1_mse = disp_array_1[4:rows_img_1-4, 4:cols_img_1-4]

	mse_disp1 = (np.sum((ground_truth_1-disp_array_1_mse)**2))/(rows_img_1*cols_img_1)
	print "mse value for view1 for 9*9 is ",mse_disp1
	
	# Getting the disparity matrix for view5
	for i in range(4, rows_img_5-4):
		for j in range(4,cols_img_5-4):

			sub_matrix_5 = padded_image_5[i-4:i+5,j-4:j+5]
			min_arr = []
			for k in range(j,j+75):
				if (k >= cols_img_5-4 ):
					k=cols_img_5-5
				sub_matrix_1 = padded_image_1[i-4:i+5,k-4:k+5]

				subtract_matrix = np.subtract(sub_matrix_1, sub_matrix_5)
				SSD_value = np.sum(np.square(subtract_matrix))
				min_arr.append([SSD_value,k])

			min_value = min(min_arr)
			disp = abs(j - (min_value[1]))
			disp_array_5[i][j] = disp


	disp_img5 = disp_array_5/disp_array_5.max()
	cv2.imshow('Disparity map 2', disp_img5)
	cv2.waitKey(0)

	disp_array_5_mse = disp_array_5[4:rows_img_5-4, 4:cols_img_5-4]
	mse_disp5 = (np.sum((ground_truth_5-disp_array_5_mse)**2))/(rows_img_5*cols_img_5)
	print "mse value for view5 for 9*9 is ",mse_disp5
	
	# Consistency check
	disp_array_consistency_1 = np.zeros((rows_img_1,cols_img_1))
	disp_array_consistency_5 = np.zeros((rows_img_1,cols_img_1))

	
	for i in range(4,rows_img_5-4):
		for j in range(4,cols_img_5-4):

			# if (disp_array_1[i][j] != disp_array_5[i][abs(j - int(disp_array_1[i][j]))]) and (disp_array_5[i][j] != disp_array_1[i][abs(j + int(disp_array_5[i][j]))]):
			if (abs(j - int(disp_array_1[i][j])) >= 0):
				if (disp_array_1[i][j] == disp_array_5[i][abs(j - int(disp_array_1[i][j]))]):
					disp_array_consistency_1[i][j] = disp_array_1[i][j]
				else:
					disp_array_consistency_1[i][j] = 0

			if (abs(j - int(disp_array_5[i][j])) >= 0):
				if (disp_array_5[i][j] == disp_array_1[i][abs(j + int(disp_array_5[i][j]))]):
					disp_array_consistency_5[i][j] = disp_array_5[i][j]
				else:
					disp_array_consistency_5[i][j] = 0


	disp_img_consistency_1 = disp_array_consistency_1/disp_array_consistency_1.max()
	disp_img_consistency_5 = disp_array_consistency_5/disp_array_consistency_5.max()
	cv2.imshow('Disparity map for left image after consistency check for 9*9 block', disp_img_consistency_1)
	cv2.waitKey(0)
	cv2.imshow('Disparity map for right image after consistency check for 9*9 block', disp_img_consistency_5)
	cv2.waitKey(0)

	#disp_array_consistency = disp_array_consistency[1:rows_img_1-1, 1:cols_img_1-1]

	consistency_arr_1 = []
	consistency_arr_5 = []

	for i in range(4,rows_img_5-4):
		for j in range(4,cols_img_5-4):
			if (int(disp_array_consistency_1[i][j]) != 0) and ((int(disp_array_consistency_5[i][j]) != 0)) and (i < 370 and j < 463):
				
				consistency_arr_1.append(ground_truth_1[i][j]-disp_array_consistency_1[i][j])
				consistency_arr_5.append(ground_truth_5[i][j]-disp_array_consistency_5[i][j])

	consistency_arr_1 = np.array(consistency_arr_1)
	consistency_arr_5 = np.array(consistency_arr_5)

	mse_disp1_consistency = (np.sum(consistency_arr_1**2))/(rows_img_1*cols_img_1)
	mse_disp5_consistency = (np.sum(consistency_arr_5**2))/(rows_img_5*cols_img_5)

	print "mse_disp1_consistency =", mse_disp1_consistency
	print "mse_disp5_consistency =", mse_disp5_consistency

def view_synthesis():
	img_1 = cv2.imread('view1.png')
	img_5 = cv2.imread('view5.png')

	ground_truth_left_disp = cv2.imread('disp1.png',0)
	ground_truth_right_disp = cv2.imread('disp5.png',0)

	height,width,depth = img_1.shape

	synthesized_left_image = np.zeros((height,width,depth),dtype = np.uint8)
	synthesized_right_image = np.zeros((height,width,depth),dtype = np.uint8)

	for i in range(0, height):
		for j in range(0,width):
			for k in range(0,depth):

				index_left = j - ground_truth_left_disp[i][j]/2
				index_right = j + ground_truth_right_disp[i][j]/2

				synthesized_left_image[i][index_left][k] = img_1[i][j][k] 

				if index_right < 463:
					synthesized_right_image[i][index_right][k] = img_5[i][j][k] 

	new_rows,new_cols,new_depth = synthesized_left_image.shape

	for i in range(0,new_rows):
		for j in range(0,new_cols):
			for k in range(0,new_depth):
				if (int(synthesized_left_image[i][j][0]) == 0) and (int(synthesized_left_image[i][j][1]) == 0) and (int(synthesized_left_image[i][j][2]) == 0):
						synthesized_left_image[i][j][0] = synthesized_right_image[i][j][0]
						synthesized_left_image[i][j][1] = synthesized_right_image[i][j][1]
						synthesized_left_image[i][j][2] = synthesized_right_image[i][j][2]


	cv2.imshow("View synthesized image", synthesized_left_image)
	#cv2.imwrite("viewSyn_right.png", synthesized_right_image)
	cv2.waitKey(0)

#three_cross_three_block()
view_synthesis()
#nine_cross_nine_block()

print 'Execution finished in ' + str(round(time.time() - start_time, 2)) + 's'
