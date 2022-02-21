#pragma once
//���o�́A�p�����[�^�̎w��A�p�b�`�T�C�Y�w��
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
string win_src = "src";
string win_dst = "dst";
string win_dst2 = "dst2";

// �p�����[�^�w��(L�ARamba)
#define L 2			// �s���~�b�h���x��(0~L)
int width;
int height;
double fps;			// �t���[����
int frame;			// �擾�t���[����
double Rambda = 150;	// �F�l�ƃe�N�X�`���̔䗦 Ramba
double r_max;			// the maximum dimention (of video��)
double Ro = 0.5;		// the reduction fuctor of the search window
int RandUniform = 1;

int MAX_DATA;			// ���s�N�Z����
int MAX_INTENSE = 255;	// �ő�F�l
double Ramda = 0;		// �f�[�^�����i��������10^(-7)�j
double Alpha = 0.001;	// �����������i�������p�����[�^�j
double Sigma = 32;		// �m�C�Y���x���i�W���΍��j
double Mean = 0;		// �m�C�Y���x���i���ρj
double Converge = 1.0e-10;	// ��������l
int Repeat = 10000;			// �ő唽����

// �p�b�`
#define PATCHSIZE 5
int PATCHSIZEint = 5;
int PATCHSIZEint2 = 25;
int PATCHSIZEint3 = 125;
int PATCHstart = -2;
int PATCHend = 3;

Mat video[200];	// ���摜�̉摜�W��
Mat orig_video[200];
Mat img_mask;	// �}�X�N�摜�i�O���[�X�P�[���j�̓ǂݍ���
Mat img_src;	// ���͉摜�i�J���[�j�̓ǂݍ���
Mat img_src2;	// ���͉摜�i�O���[�X�P�[���j�̓ǂݍ���
Mat img_SRC;	// ��C����摜�i�J���[�j
Mat img_SRC2;	// ��C����摜�i�O���[�X�P�[���j
Mat img_dst;	// �o�͉摜�i�J���[�j�̐ݒ�
Mat img_dst3;
Mat img_dst2[200];	// MRF�K����o�͉摜�i�J���[�j

// ���摜�iColorImage, TextureFeature, Occlusion�S�Ή��j
class Video {
private:
	Mat Image_temp;
public:
	int XSIZE;
	int YSIZE;
	int ZSIZE = frame;		// Frame numbers;
	int Channel;		// Channnel(3:ColorImage, 2:TextureFeature, 1:Occlusion)
	vector<Mat> Image;	// ImageSet

	Video();				// ���摜�̏�����
	Video(int, int, int);	// ���摜�̌^������
	void push_back(Mat&);	// �摜�W�����瓮����擾
	//void copyTo(Video&);	// ���摜�̕���
	void setToGray();
};
Video::Video() {
	Image.clear();
}
Video::Video(int VideoXsize, int VideoYsize, int VideoChannel) {
	XSIZE = VideoXsize;
	YSIZE = VideoYsize;
	ZSIZE = frame;
	Channel = VideoChannel;
	Image.clear();
}
void Video::push_back(Mat& video_Image) {
	Image.push_back(video_Image);
}
//void Video::copyTo(Video& video_dst) {	// ���������R�s�[���Ȃ����ߎg�p�s��(VideoCopy���g��)
//	video_dst = Video(XSIZE, YSIZE, Channel);
//	for (int videoZ = 0; videoZ < frame; videoZ++) {
//		Image[videoZ].copyTo(Image_temp);
//		video_dst.push_back(Image_temp);
//		//if (videoZ == 0) { video_dst.Image[videoZ].copyTo(img_SRC); }	// �m�F�p
//	}
//}
void Video::setToGray() {
	Image_temp.create(YSIZE, XSIZE, CV_8UC1);
	uchar gray = (uchar)0;
	Image_temp.setTo(gray);
	if (Channel == 1) {
		for (int videoZ = 0; videoZ < frame; videoZ++) {
			Image.push_back(Image_temp);
		}
	}
	else { cout << "ERROR: Video Set To GRAY" << endl; }
}


// Video copy function
void VideoCopy(Video& src, Video& dst) {
	dst = Video(src.XSIZE, src.YSIZE, src.Channel);
	for (int tt = 0; tt < frame; tt++) {
		dst.push_back(src.Image[tt]);
	}
}
void VideoCopyTo(Video& src, Video& dst) {
	Mat tempImg;
	dst = Video(src.XSIZE, src.YSIZE, src.Channel);
	for (int tt = 0; tt < frame; tt++) {
		tempImg = src.Image[tt].clone();
		dst.push_back(tempImg);
	}
}

int unoccluded_checker(Point3i, Video&);				// count the number of occluded pixels 

// ���摜�W���i�C�ӂ̑w�ł̓��摜�j
class VideoSet {
private:
	Mat video_temp;
public:
	int X_SIZE;
	int Y_SIZE;
	int Z_SIZE = frame;
	Video Color;		// ColorImage
	Video Occlusion;	// Occlusion
	Video Texture;		// TextureFeature
	int P_LEVEL;

	VideoSet();					// ���摜�W���̏�����
	VideoSet(int, int);			// ���摜�̌^������
	//void push(int, Video&);		// �摜�W�����瓮����擾
	void push_all(Video&, Video&, Video&);
	void copyTo(VideoSet&);
	void copyLevel(int);
};
VideoSet::VideoSet() {
	Color = Video();
	Occlusion = Video();
	Texture = Video();
}
VideoSet::VideoSet(int Video_Xsize, int Video_Ysize) {
	X_SIZE = Video_Xsize;
	Y_SIZE = Video_Ysize;
	Z_SIZE = frame;
}
//void VideoSet::push(int Set_Index, Video& Set_Image) {
//	int set_number = Set_Index;
//	switch (set_number)
//	{
//	case(0):
//		Set_Image.copyTo(Color);		// ��C���铮�摜�쐬�i�J���[�j
//		cout << "input Color Image" << endl;
//	case(1):
//		Set_Image.copyTo(Occlusion);	// ��C�̈�w��摜
//		cout << "input Occlusion Image" << endl;
//	case(2):
//		Set_Image.copyTo(Texture);	// ��C���铮�摜�쐬�i�O���[�X�P�[���j
//		cout << "input Texture Image" << endl;
//	default:
//		cout << "ERROR : input Image" << endl;
//		break;
//	}
//}
void VideoSet::push_all(Video& Set_ImageU, Video& Set_ImageO, Video& Set_ImageT) {
	VideoCopy(Set_ImageU, Color);		// ��C���铮�摜�쐬�i�J���[�j
	VideoCopy(Set_ImageO, Occlusion);	// ��C�̈�w��摜
	VideoCopy(Set_ImageT, Texture);	// ��C���铮�摜�쐬�i�O���[�X�P�[���j
}
void VideoSet::copyTo(VideoSet& video_set_dst) {
	video_set_dst = VideoSet(X_SIZE, Y_SIZE);
	VideoCopy(Color, video_set_dst.Color);
	VideoCopy(Occlusion, video_set_dst.Occlusion);
	VideoCopy(Texture, video_set_dst.Texture);
}
void VideoSet::copyLevel(int Pyramid_Level) {
	P_LEVEL = Pyramid_Level;
}

// ����s���~�b�h�iColorImage, TextureFeature, Occlusion�j
class VideoPyramid {
private:
	int x_size;
	int y_size;
	int z_size = frame;
	int nLEVEL;
	VideoSet nowSet;
	Video U, O, T;
	Mat video_temp, video_temp2, video_temp3;
public:
	vector<VideoSet> nLEVEL_video;	// ColorImage, Occlusion, TextureFeature pyramid
	vector<int> nLEVEL_X_SIZE;
	vector<int> nLEVEL_Y_SIZE;

	VideoPyramid(Video&);			// make pyramid
	void output(int, VideoSet&);	// ��(int)�w�ڂ��擾
	void input(int, VideoSet&);		// ��(int)�w�ڂ��X�V
};
VideoPyramid::VideoPyramid(Video& Video_orign) {
	// �����w�ł�VideoSet�쐬(U,O)
	x_size = Video_orign.XSIZE;
	y_size = Video_orign.YSIZE;
	nowSet = VideoSet(x_size, y_size);
	O = Video(x_size, y_size, 1);		// Occlusion:�}�X�N�摜�g��
	for (int t = 0; t < frame; t++) {
		O.push_back(img_mask);
	}
	T = Video(x_size, y_size, 2);		// Texture:2�����e�N�X�`������
	Mat TextureFeature;	// �e�N�X�`�������摜
	int I_x, I_y;		// �e�N�X�`���������O���[�X�P�[���̍���
	int gray1, gray2;
	int texture_point, texture_point2;
	VideoCopy(Video_orign, U);
	//U.Image[0].copyTo(img_SRC);	// �m�F�p
	for (int t = 0; t < frame; t++) {
		U.Image[t].copyTo(video_temp);
		cvtColor(video_temp, img_SRC2, COLOR_RGB2GRAY);			// �O���[�X�P�[���ϊ�
		TextureFeature = Mat(Size(x_size, y_size), CV_8UC2);	// 2�����e�N�X�`������
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				texture_point = y * width + x;
				if (O.Image[t].data[texture_point] != 0) {
					I_x = 0; I_y = 0;	// �I�N���[�W�����摜����=>�e�N�X�`��������"0"�Ƃ���
				}
				else if (texture_point + 1 < MAX_DATA && O.Image[t].data[texture_point + 1] != 0) {
					I_x = 0; I_y = 0;
				}
				else if (texture_point + width < MAX_DATA && O.Image[t].data[texture_point + width] != 0) {
					I_x = 0; I_y = 0;
				}
				else {
					if (x < width - 1) {	// x ����
						gray1 = (int)img_SRC2.data[texture_point];
						gray2 = (int)img_SRC2.data[texture_point + 1];
						I_x = gray2 - gray1;
					}
					else { I_x = 0; }
					if (y < height - 1) {	// y ����
						gray1 = (int)img_SRC2.data[texture_point];
						gray2 = (int)img_SRC2.data[texture_point + width];
						I_y = gray2 - gray1;
					}
					else { I_y = 0; }
				}
				texture_point2 = y * width * 2 + x * 2;	// T(0)
				TextureFeature.data[texture_point2] = I_x;
				TextureFeature.data[texture_point2 + 1] = I_y;
			}
		}
		T.push_back(TextureFeature);
	}
	nowSet.push_all(U, O, T);
	nLEVEL_video.push_back(nowSet);		// nLEVEL_video[0]
	nLEVEL_X_SIZE.push_back(x_size);
	nLEVEL_Y_SIZE.push_back(y_size);
	/* �m�F */
	cout << "L = 0 , Size : " << nLEVEL_X_SIZE[0] << " : " << nLEVEL_Y_SIZE[0] << endl;


	// �e�w�ł�VideoSet�쐬
	Video img_temp_before_O, img_temp_before_U, img_temp_before_T;
	uchar gray;
	Vec3b color;
	Point2i downPoint, upPoint;
	VideoCopy(O, img_temp_before_O);
	/* �I�N���[�W�����s�N�Z���̃J�E���g(t=0�̂�) */
	int OccNUM = 0;
	for (int i = 0; i < img_temp_before_O.XSIZE * img_temp_before_O.YSIZE; i++) {
		if (img_temp_before_O.Image[0].data[i] != 0) { OccNUM++; }
	}
	cout << "Occlusion number in 0 (t=0) : " << OccNUM << endl;
	VideoCopy(U, img_temp_before_U);
	VideoCopy(T, img_temp_before_T);

	for (int pyrLevel = 1; pyrLevel <= L; pyrLevel++) {
		// VideoSet������
		if (x_size % 2 == 0) { x_size = x_size / 2; }
		else { x_size = (x_size + 1) / 2; }
		if (y_size % 2 == 0) { y_size = y_size / 2; }
		else { y_size = (y_size + 1) / 2; }

		nowSet = VideoSet();
		nowSet = VideoSet(x_size, y_size);
		nLEVEL_X_SIZE.push_back(x_size);
		nLEVEL_Y_SIZE.push_back(y_size);
		/* �m�F */
		cout << "L = " << pyrLevel << " , Size : " << nLEVEL_X_SIZE[pyrLevel] << " : " << nLEVEL_Y_SIZE[pyrLevel] << endl;
	
		OccNUM = 0;
		O = Video(x_size, y_size, 1);
		T = Video(x_size, y_size, 2);
		U = Video(x_size, y_size, 3);
		
		for (int t = 0; t < frame; t++) {
			video_temp = Mat(y_size, x_size, CV_8UC1);
			gray = (uchar)0;
			video_temp.setTo(gray);
			video_temp2 = Mat(y_size, x_size, CV_8UC2);
			video_temp3 = Mat(y_size, x_size, CV_8UC3);
			for (int y = 0; y < y_size; y++) {
				for (int x = 0; x < x_size; x++) {
					downPoint = Point2i(x, y);
					upPoint = Point2i(x * 2, y * 2);	// �����s��̍폜�i�P�n�܂�j

					// Occlusion pyramid�̍쐬 : �P���k���i�S���P�ł���C�̈悪����΃I�N���[�W�����j
					gray = (uchar)img_temp_before_O.Image[t].data[upPoint.y * img_temp_before_O.XSIZE + upPoint.x];
					if (gray != 0) {
						gray = (uchar)255;
						video_temp.data[downPoint.y * x_size + downPoint.x] = gray;
						if (t == 0) { OccNUM++; }
					}

					// Color Image pyramid�̍쐬 : �P���k���i������̍폜�j
					color = img_temp_before_U.Image[t].at<Vec3b>(upPoint.y, upPoint.x);
					video_temp3.at<Vec3b>(downPoint.y, downPoint.x) = color;

					// TectureFeature pyramid�̍쐬 : �e�w�ł̃e�N�X�`�����v�Z�i�_�E���T���v�����O�j
					texture_point = downPoint.y * x_size * 2 + downPoint.x * 2;
					texture_point2 = upPoint.y * img_temp_before_T.XSIZE * 2 + upPoint.x * 2;
					I_x = img_temp_before_T.Image[t].data[texture_point2];
					I_y = img_temp_before_T.Image[t].data[texture_point2 + 1];
					I_x = I_x * 2;
					I_y = I_y * 2;
					video_temp2.data[texture_point] = I_x;
					video_temp2.data[texture_point + 1] = I_y;
				}
			}
			O.push_back(video_temp);
			T.push_back(video_temp2);
			U.push_back(video_temp3);
		}
		nowSet.push_all(U, O, T);
		nLEVEL_video.push_back(nowSet);		// nLEVEL_video[L]
		img_temp_before_O = Video();
		VideoCopy(O, img_temp_before_O);
		img_temp_before_U = Video();
		VideoCopy(U, img_temp_before_U);
		img_temp_before_T = Video();
		VideoCopy(T, img_temp_before_T);
		/* �I�N���[�W�����s�N�Z���̃J�E���g(�\��)*/
		cout << "Occlusion number in " << pyrLevel << " (t=0) : " << OccNUM << endl;
	}
	img_temp_before_O = Video();
	img_temp_before_U = Video();
	img_temp_before_T = Video();
}
void VideoPyramid::output(int PYRAMID_LEVEL, VideoSet& L_XideoSet) {
	nLEVEL_video[PYRAMID_LEVEL].copyTo(L_XideoSet);
	L_XideoSet.copyLevel(PYRAMID_LEVEL);
}
void VideoPyramid::input(int PYRAMID_LEVEL, VideoSet& L_XideoSet) {
	L_XideoSet.copyTo(nLEVEL_video[PYRAMID_LEVEL]);
}

// �V�t�g�}�b�v�i���΃x�N�g���j���
class ShiftMap {
private:
	int point, pt;
	Point2i p2;
	Point3i p3;
public:
	int shift_level;
	int xsize;
	int ysize;
	int zsize;
	vector<Point3i> shift;	// �V�t�g�}�b�v
	//vector<Point2i> shift;
	//vector<int> shift_t;

	ShiftMap(VideoSet&);				// �����V�t�g�}�b�v (ramdom)
	Point3i nn(Point2i&, int&);			// �V�t�g�}�b�v�Ăяo��
	int nnX(Point2i&, int&);
	int nnY(Point2i&, int&);
	int nnT(Point2i&, int&);
	void put(Point3i&, Point3i&);	// �V�t�g�}�b�v�ύX
	void upsample(Video&);			// upsampling Shift map
};
ShiftMap::ShiftMap(VideoSet& lclass) {
	shift_level = lclass.P_LEVEL;
	xsize = lclass.X_SIZE;
	ysize = lclass.Y_SIZE;
	zsize = lclass.Z_SIZE;
	for (int h = 0; h < zsize; h++) {
		for (int i = 0; i < ysize; i++) {
			for (int j = 0; j < xsize; j++) {
				p2 = Point2i(j, i);
				pt = h;
				while (lclass.Occlusion.Image[pt].data[p2.y * xsize + p2.x] != 0) {
					p2.x = rand() % xsize;	// ��C�̈�̏����V�t�g�}�b�v (ramdom)
					p2.y = rand() % ysize;
					pt = rand() % zsize;
					/*cout << p2 << ":" << pt << " -> " << (int)lclass.Occlusion.Image[pt].data[p2.y * lclass.X_SIZE + p2.x] << endl;*/
				}
				p3 = Point3i(p2.x - j, p2.y - i, pt - h);
				shift.push_back(p3);
				/*cout << Point3i(j,i,h) << endl;*/
			}
		}
	}
	p3 = Point3i(0, 0, 0);
	for (int k = xsize * ysize * zsize; k < width * height * frame; k++) {
		shift.push_back(p3);
	}
	///* �����V�t�g�}�b�v�m�F */
	//cout << "�����V�t�g�}�b�v" << endl;
	//for (int i = 0; i < xsize * ysize * zsize; i++) {
	//	cout << shift[i] << endl;
	//}
}
Point3i ShiftMap::nn(Point2i& POINT, int& TIME) {
	point = TIME*xsize*ysize + POINT.y * xsize + POINT.x;
	return shift[point];
}
int ShiftMap::nnX(Point2i& POINT, int& TIME) {
	point = TIME * xsize * ysize + POINT.y * xsize + POINT.x;
	return shift[point].x;
}
int ShiftMap::nnY(Point2i& POINT, int& TIME) {
	point = TIME * xsize * ysize + POINT.y * xsize + POINT.x;
	return shift[point].y;
}
int ShiftMap::nnT(Point2i& POINT, int& TIME) {
	point = TIME * xsize * ysize + POINT.y * xsize + POINT.x;
	return shift[point].z;
}
void ShiftMap::put(Point3i& POINT, Point3i& NNpoint) {
	point = POINT.z * ysize * xsize + POINT.y * xsize + POINT.x;
	shift[point] = NNpoint;
}
void ShiftMap::upsample(Video& NEWoccl) {
	vector<Point3i> temp;
	Point3i tempP;
	int temp_ind;
	for (int numT = 0; numT < zsize; numT++) {
		for (int num = 0; num < xsize * ysize; num++) {
			temp_ind = numT * xsize * ysize + num;
			tempP = Point3i(shift[temp_ind].x * 2, shift[temp_ind].y * 2, shift[temp_ind].z);
			temp.push_back(tempP);
		}
	}
	xsize = 2 * xsize;
	ysize = 2 * ysize;
	shift_level--;
	shift.clear();

	// �V�t�g�}�b�v�S�_�̂�����C�̈�͈��p��
	int before, recheck_count = 0;
	Point2i check, now;
	int check_t, now_t;
	vector<Point3i> recheck;
	int occ_checker;
	for (int h = 0; h < zsize; h++) {
		now_t = h;
		for (int i = 0, countY = 0; i < ysize; i++) {
			for (int j = 0, countX = 0; j < xsize; j++) {
				p2 = Point2i(0, 0);
				now = Point2i(j, i); 
				tempP = Point3i(j, i, h);
				occ_checker = unoccluded_checker(tempP, NEWoccl);
				if (occ_checker != 0)/*if (NEWoccl.Image[h].data[i * NEWoccl.XSIZE + j] != 0)*/ {
					before = h * (ysize / 2) * (xsize / 2) + countY * (xsize / 2) + countX;
					check = Point2i(j + temp[before].x, i + temp[before].y);
					check_t = h + temp[before].z;
					if (check.x > 0 && check.x < NEWoccl.XSIZE && check.y > 0 && check.y < NEWoccl.YSIZE && check_t > 0 && check_t < NEWoccl.ZSIZE) {
						if (NEWoccl.Image[check_t].data[check.y * NEWoccl.XSIZE + check.x] != 0) {
							if (NEWoccl.Image[check_t].data[check.y * NEWoccl.XSIZE + (check.x - 1)] != 0 || check.x - 1 < 0) {
								if (NEWoccl.Image[check_t].data[(check.y - 1) * NEWoccl.XSIZE + check.x] != 0 || check.y - 1 < 0) {
									if (NEWoccl.Image[check_t].data[(check.y - 1) * NEWoccl.XSIZE + (check.x - 1)] != 0 || check.x - 1 < 0 || check.y - 1 < 0) {
										Point3i errorP = Point3i(now.x, now.y, now_t);
										recheck.push_back(errorP);
										recheck_count++;
										cout << "Shift Upsamplig ERROR! :" << errorP << " : " << check << check_t << endl;
									}
									else { before = before - NEWoccl.XSIZE - 1; }
								}
								else { before = before - NEWoccl.XSIZE; }
							}
							else { before = before - 1; }
						}
					}
					//else { cout << "Shift Upsamplig ERROR! : size" << endl; }
					p2 = Point2i(temp[before].x, temp[before].y);
				}
				p3 = Point3i(p2.x, p2.y, h);
				shift.push_back(p3);
				if ((j + 1) % 2 == 0) { countX++; }
			}
			if ((i + 1) % 2 == 0) { countY++; }
		}
	}
	temp.clear();
	recheck.clear();

	///* �V�t�g�}�b�v�m�F */
	//cout << shift_level << " :�V�t�g�}�b�v" << endl;
	//for (int h = 0; h < zsize; h++) {
	//	for (int i = 0; i < ysize; i++) {
	//		for (int j = 0; j < xsize; j++) {
	//			cout << "[" << j << ", " << i << ", " << h << "] -> " << shift[h * ysize * xsize + i * xsize + j];
	//			if (NEWoccl.Image[h].data[i * NEWoccl.XSIZE + j] != 0) {
	//				cout << "   ###" << endl;
	//			}
	//			else { cout << endl; }
	//		}
	//	}
	//}
}

// �e�p�b�`���
class Patch {
private:
	int start_value = PATCHstart;
	int end_value = PATCHSIZEint - start_value;
public:
	int patchSizeX = PATCHSIZEint;
	int patchSizeY = PATCHSIZEint;
	int patchSizeZ = PATCHSIZEint;
	int PatchSize = patchSizeX * patchSizeY * patchSizeZ;
	int hPatchSize;				// ���C�̈�p�b�`��
	int nElsTotal;				// �̈�O�p�b�`��
	vector<Point3i> NN;			// �V�t�g�}�b�v
	vector<Point3i> ANN;
	vector<Vec3b> ColorNN;		// �F�l(BGR)
	vector<Vec3b> ColorANN;
	vector<float> IxNN;			// �e�N�X�`������
	vector<float> IxANN;
	vector<float> IyNN;
	vector<float> IyANN;
	vector<int> OcclusionChecker; // ��C�̈�m�F�l

	Patch();													// ������
	Patch(int, int, int, Video&, Video&, Video&, ShiftMap&);	// �C�ӂ̓_�̃R�X�g�֐�
	Patch(Point3i, Point3i, Video&, Video&, Video&, ShiftMap&);	// �Q�_��r���̃R�X�g�֐�
	double costfunction(int);	// �R�X�g�֐��̌v�Z(0:�I�N���[�W�����Ȃ�d, 1:�I�N���[�W�����Ȃ�d^2, 2:�I�N���[�W�����܂�d, 3:�I�N���[�W�����܂�d^2)
};
Patch::Patch() {
	hPatchSize = 0;
	nElsTotal = 0;
	NN.clear();
	ANN.clear();
	ColorNN.clear();
	ColorANN.clear();
	IxNN.clear();
	IxANN.clear();
	IyNN.clear();
	IyANN.clear();
	OcclusionChecker.clear();
}
Patch::Patch(int X_p, int Y_p, int Z_p, Video& U_p, Video& T_p, Video& O_p, ShiftMap& sm_p) {
	Point2i XY_p = Point2i(X_p, Y_p);
	for (int ppZ = Z_p + start_value; ppZ < Z_p + end_value; ppZ++) {
		for (int ppY = Y_p + start_value; ppY < Y_p + end_value; ppY++) {
			for (int ppX = X_p + start_value; ppX < X_p + end_value; ppX++) {
				NN.push_back(Point3i(ppX, ppY, ppZ));
				ANN.push_back(Point3i(ppX + sm_p.nnX(XY_p, Z_p), ppY + sm_p.nnY(XY_p, Z_p), ppZ + sm_p.nnT(XY_p, Z_p)));
			}
		}
	}
	hPatchSize = 0;	// ���C�̈�p�b�`��
	nElsTotal = 0;	// �̈�O�p�b�`��
	for (int pp = 0; pp < PatchSize; pp++) {
		if (NN[pp].x < 0 || NN[pp].x >= U_p.XSIZE || NN[pp].y < 0 || NN[pp].y >= U_p.YSIZE || NN[pp].z < 0 || NN[pp].z >= U_p.ZSIZE) {
			OcclusionChecker.push_back(2);
			nElsTotal++;
		}
		else if (ANN[pp].x < 0 || ANN[pp].x >= U_p.XSIZE || ANN[pp].y < 0 || ANN[pp].y >= U_p.XSIZE || ANN[pp].z < 0 || ANN[pp].z >= U_p.ZSIZE) {
			OcclusionChecker.push_back(2);
			nElsTotal++;
		}
		else {
			if (O_p.Image[ANN[pp].z].at<uchar>(ANN[pp].y, ANN[pp].x) != 0) {
				OcclusionChecker.push_back(-1);
				hPatchSize++;
			}
			else if (O_p.Image[NN[pp].z].at<uchar>(NN[pp].y, NN[pp].x) != 0) {
				OcclusionChecker.push_back(1);
				hPatchSize++;
			}
			else { OcclusionChecker.push_back(0); }
			ColorNN.push_back(U_p.Image[NN[pp].z].at<Vec3b>(NN[pp].y, NN[pp].x));
			ColorANN.push_back(U_p.Image[ANN[pp].z].at<Vec3b>(ANN[pp].y, ANN[pp].x));
			IxNN.push_back((int)(T_p.Image[NN[pp].z].data[NN[pp].y * T_p.XSIZE * 2 + NN[pp].x * 2 + 0]));
			IyNN.push_back((int)(T_p.Image[NN[pp].z].data[NN[pp].y * T_p.XSIZE * 2 + NN[pp].x * 2 + 1]));
			IxANN.push_back((int)(T_p.Image[ANN[pp].z].data[ANN[pp].y * T_p.XSIZE * 2 + ANN[pp].x * 2 + 0]));
			IyANN.push_back((int)(T_p.Image[ANN[pp].z].data[ANN[pp].y * T_p.XSIZE * 2 + ANN[pp].x * 2 + 1]));
		}
	}
}
Patch::Patch(Point3i P_1, Point3i P_2, Video& U_p, Video& T_p, Video& O_p, ShiftMap& sm_p) {
	for (int ppZ = start_value; ppZ < end_value; ppZ++) {
		for (int ppY = start_value; ppY < end_value; ppY++) {
			for (int ppX = start_value; ppX < end_value; ppX++) {
				NN.push_back(Point3i(P_1.x + ppX, P_1.y + ppY, P_1.z + ppZ));
				ANN.push_back(Point3i(P_2.x + ppX, P_2.y + ppY, P_2.z + ppZ));
			}
		}
	}
	hPatchSize = 0;
	nElsTotal = 0;
	for (int pp = 0; pp < PatchSize; pp++) {
		if (NN[pp].x < 0 || NN[pp].x >= U_p.XSIZE || NN[pp].y < 0 || NN[pp].y >= U_p.YSIZE || NN[pp].z < 0 || NN[pp].z >= U_p.ZSIZE) {
			OcclusionChecker.push_back(2);
			nElsTotal++;
		}
		else if (ANN[pp].x < 0 || ANN[pp].x >= U_p.XSIZE || ANN[pp].y < 0 || ANN[pp].y >= U_p.YSIZE || ANN[pp].z < 0 || ANN[pp].z >= U_p.ZSIZE) {
			OcclusionChecker.push_back(2);
			nElsTotal++;
		}
		else {
			//cout << "NN: " << P_1 << " ,  ANN: " << P_2 << endl;	// �m�F�p
			if (O_p.Image[ANN[pp].z].at<uchar>(ANN[pp].y, ANN[pp].x) != 0) {
				OcclusionChecker.push_back(-1);
				hPatchSize++;
			}
			if (O_p.Image[NN[pp].z].at<uchar>(NN[pp].y, NN[pp].x) != 0) {
				OcclusionChecker.push_back(1);
				hPatchSize++;
			}
			else { OcclusionChecker.push_back(0); }
			ColorNN.push_back(U_p.Image[NN[pp].z].at<Vec3b>(NN[pp].y, NN[pp].x));
			ColorANN.push_back(U_p.Image[ANN[pp].z].at<Vec3b>(ANN[pp].y, ANN[pp].x));
			IxNN.push_back((int)(T_p.Image[NN[pp].z].data[NN[pp].y * T_p.XSIZE * 2 + NN[pp].x * 2 + 0]));
			IyNN.push_back((int)(T_p.Image[NN[pp].z].data[NN[pp].y * T_p.XSIZE * 2 + NN[pp].x * 2 + 1]));
			IxANN.push_back((int)(T_p.Image[ANN[pp].z].data[ANN[pp].y * T_p.XSIZE * 2 + ANN[pp].x * 2 + 0]));
			IyANN.push_back((int)(T_p.Image[ANN[pp].z].data[ANN[pp].y * T_p.XSIZE * 2 + ANN[pp].x * 2 + 1]));
		}
	}
}
double Patch::costfunction(int info) {
	int Total = PatchSize - nElsTotal - hPatchSize;
	int diff, sumU = 0;
	int sumTx = 0, sumTy = 0;
	double cardNN = 2.0, cardANN = 2.0;
	double answer;
	switch (info) {
	case 0:
		for (int pp = 0; pp < Total; pp++) {
			if (OcclusionChecker[pp] == 0) {
				// Cost sum of U
				for (int channel = 0; channel < 3; channel++) {
					diff = (int)ColorNN[pp][channel] - (int)ColorANN[pp][channel];
					diff = diff * diff;
					sumU = sumU + diff;
				}
				// Cost sum of T
				sumTx = sumTx + (pow((IxNN[pp] - IxANN[pp]), 2) / cardNN);
				sumTy = sumTy + (pow((IyNN[pp] - IyANN[pp]), 2) / cardANN);
			}
		}
		/*sumTx = sumTx + sumTy;*/
		sumTx = sqrt(sumTx) + sqrt(sumTy);
		sumU = sqrt(sumU);
		sumTx = sqrt(sumTx);
		break;
	case 1:
		for (int pp = 0; pp < Total; pp++) {
			if (OcclusionChecker[pp] == 0) {
				// Cost sum of U
				for (int channel = 0; channel < 3; channel++) {
					diff = (int)ColorNN[pp][channel] - (int)ColorANN[pp][channel];
					diff = diff * diff;
					sumU = sumU + diff;
				}
				// Cost sum of T
				sumTx = sumTx + (pow((IxNN[pp] - IxANN[pp]), 2) / cardNN);
				sumTy = sumTy + (pow((IyNN[pp] - IyANN[pp]), 2) / cardANN);
			}
		}
		/*sumTx = sumTx + sumTy;*/
		sumTx = sqrt(sumTx) + sqrt(sumTy);
		break;
	case 2:
		for (int pp = 0; pp < Total; pp++) {
			if (OcclusionChecker[pp] != 2 && OcclusionChecker[pp] != -1) {
				// Cost sum of U
				for (int channel = 0; channel < 3; channel++) {
					diff = (int)ColorNN[pp][channel] - (int)ColorANN[pp][channel];
					diff = diff * diff;
					sumU = sumU + diff;
				}
				// Cost sum of T
				sumTx = sumTx + (pow((IxNN[pp] - IxANN[pp]), 2) / cardNN);
				sumTy = sumTy + (pow((IyNN[pp] - IyANN[pp]), 2) / cardANN);
			}
		}
		/*sumTx = sumTx + sumTy;*/
		sumTx = sqrt(sumTx) + sqrt(sumTy);
		sumU = sqrt(sumU);
		sumTx = sqrt(sumTx);
		break;
	case 3:
		for (int pp = 0; pp < Total; pp++) {
			if (OcclusionChecker[pp] != 2 && OcclusionChecker[pp] != -1) {
				// Cost sum of U
				for (int channel = 0; channel < 3; channel++) {
					diff = (int)ColorNN[pp][channel] - (int)ColorANN[pp][channel];
					diff = diff * diff;
					sumU = sumU + diff;
				}
				// Cost sum of T
				sumTx = sumTx + (pow((IxNN[pp] - IxANN[pp]), 2) / cardNN);
				sumTy = sumTy + (pow((IyNN[pp] - IyANN[pp]), 2) / cardANN);
			}
		}
		/*sumTx = sumTx + sumTy;*/
		sumTx = sqrt(sumTx) + sqrt(sumTy);
		break;
	default:
		cout << "ERROR: fail calculate cost function." << endl;
	}
	answer = (sumU + (double)Rambda * sumTx) / (double)Total;
	//if (Total < 9) { answer = 1000000.0; }
	if (Total < 9 && answer == 0) { answer = 1000000.0; }
	//if (answer == 0) { cout << "! answer == 0  &  Total = " << Total << endl; }
	if (info == 0) { answer = sqrt(answer); }
	return answer;
}

// ��������_�}�b�`���O�ɂ��V�t�g���
class VideoShift {
private:
	int point, pt;
	Point2i p2;
	Point3i p3;
public:
	int shift_level;
	int xsize;
	int ysize;
	int zsize;
	vector<Mat> v_shift;	// �V�t�g�}�b�v
	vector<Point2i> V_shift;

	VideoShift(Video&, int, int);
	VideoShift(Video&, Video&, int, int);
	void changeLEVEL(int);
	void clear();
};
VideoShift::VideoShift(Video& video_L, int pyramid_L, int featureLEVEL) {
	shift_level = pyramid_L;
	xsize = video_L.XSIZE;
	ysize = video_L.YSIZE;
	zsize = frame;

	Mat src1, src2;
	Mat mask1, mask2;
	for (int h = 0; h < zsize - 1; h++) {
		//cout << "h: " << h << endl;	// �m�F�p
		// �����_�}�b�`���O�摜�̑I�o
		video_L.Image[h].copyTo(src1);
		video_L.Image[h + 1].copyTo(src2);

		// �����_�̒��o�ƋL�q
		vector<KeyPoint> keypoint1, keypoint2;	// �����_(�L�[�|�C���g)
		Mat descriptor1, descriptor2;			// ������(�f�B�X�N���v�^)
		Ptr<ORB>  feature = ORB::create();		// ORB�I�u�W�F�N�g�쐬
		feature->detectAndCompute(src1, noArray(), keypoint1, descriptor1);
		feature->detectAndCompute(src2, noArray(), keypoint2, descriptor2);

		// �����_�}�b�`���O
		vector<DMatch> allMatch, goodMatch;
		BFMatcher matcher(NORM_HAMMING);
		matcher.match(descriptor1, descriptor2, allMatch);

		// ���Ă�������_�̂݃s�b�N�A�b�v
		for (int i = 0; i < (int)allMatch.size(); i++) {
			if (allMatch[i].distance < featureLEVEL) goodMatch.push_back(allMatch[i]);
		}

		// �`��
		Mat V_dst;
		drawMatches(src1, keypoint1, src2, keypoint2, goodMatch, V_dst);
		//if (h == 0) { V_dst.copyTo(img_SRC); }	// �m�F�p

		// ��v��
		float matchRate = (float)goodMatch.size() / (float)keypoint1.size();
		//cout << "��v��: " << matchRate << endl;	// �m�F�p
		//cout << "keypoint1: " << (int)keypoint1.size() << endl;
		//cout << "keypoint2: " << (int)keypoint2.size() << endl;
		//cout << "allMatch: " << (int)allMatch.size() << endl;
		//cout << "goodMatch: " << (int)goodMatch.size() << endl;

		// �����_�Ή��t��
		v_shift.push_back(V_dst);

		int index1, index2, shift_aveX = 0, shift_aveY = 0, occ_point = 0;
		double s_aX, s_aY;
		for (int i = 0; i < (int)goodMatch.size(); i++) {
			index1 = keypoint1[goodMatch[i].queryIdx].pt.y * img_mask.cols + keypoint1[goodMatch[i].queryIdx].pt.x;
			index2 = keypoint2[goodMatch[i].trainIdx].pt.y * img_mask.cols + keypoint2[goodMatch[i].trainIdx].pt.x;
			if (img_mask.data[index1] != 0 || img_mask.data[index2] != 0) { occ_point++; }
			else {
				shift_aveX = shift_aveX + (keypoint2[goodMatch[i].trainIdx].pt.x - keypoint1[goodMatch[i].queryIdx].pt.x);
				shift_aveY = shift_aveY + (keypoint2[goodMatch[i].trainIdx].pt.y - keypoint1[goodMatch[i].queryIdx].pt.y);
				//cout << "   " << goodMatch[i].queryIdx << " : " << goodMatch[i].trainIdx << endl;	// �m�F�p
			}
		}
		s_aX = (double)shift_aveX / ((int)goodMatch.size() - occ_point);
		s_aY = (double)shift_aveY / ((int)goodMatch.size() - occ_point);
		shift_aveX = (int)round(s_aX), shift_aveY = (int)round(s_aY);
		V_shift.push_back(Point2i(shift_aveX, shift_aveY));
		//cout << "shift_ave: " << Point2i(shift_aveX, shift_aveY) << endl;	// �m�F�p
		//cout << "shift_ave(X): " << s_aX/* shift_aveX*/ << endl;
		//cout << "shift_ave(Y): " << s_aY/*shift_aveY*/ << endl;
		//cout << "occ_point: " << occ_point << endl;

		/* �m�F */
		//if (shift_aveX != -1 && shift_aveY != 0) { cout << " ERROR: VideoShift point wrong!(A)" << endl; }	// A
		//if (h < 50) {																							// B
		//	if (shift_aveX != -1 && shift_aveY != 0) { cout << " ERROR: VideoShift point wrong!(B1)" << endl; }
		//}
		//else if (h < 90) {
		//	if (shift_aveX != 0 && shift_aveY != -1) { cout << " ERROR: VideoShift point wrong!(B2)" << endl; }
		//}
		//else if (h < 140) {
		//	if (shift_aveX != 1 && shift_aveY != 0) { cout << " ERROR: VideoShift point wrong!(B3)" << endl; }
		//}
		//else {
		//	if (shift_aveX != 0 && shift_aveY != 1) { cout << " ERROR: VideoShift point wrong!(B4)" << endl; }
		//}
	}
	V_shift.push_back(Point2i(0, 0));
}
VideoShift::VideoShift(Video& video_L, Video& video_OCC, int pyramid_L, int featureLEVEL) {
	shift_level = pyramid_L;
	xsize = video_L.XSIZE;
	ysize = video_L.YSIZE;
	zsize = frame;

	Mat src1, src2;
	Mat mask1, mask2;
	for (int h = 0; h < zsize - 1; h++) {
		//cout << "h: " << h << endl;	// �m�F�p
		// �����_�}�b�`���O�摜�̑I�o
		video_L.Image[h].copyTo(src1);
		video_L.Image[h + 1].copyTo(src2);
		cvtColor(src1, src1, COLOR_BGR2GRAY);	// �J���[����O���[�X�P�[���ɕύX
		cvtColor(src2, src2, COLOR_BGR2GRAY);

		// �����_�̒��o�ƋL�q
		vector<KeyPoint> keypoint1, keypoint2;	// �����_(�L�[�|�C���g)
		Mat descriptor1, descriptor2;			// ������(�f�B�X�N���v�^)
		auto  feature = AKAZE::create();	// AKAZE�I�u�W�F�N�g�쐬
		feature->detectAndCompute(src1, noArray(), keypoint1, descriptor1);
		feature->detectAndCompute(src2, noArray(), keypoint2, descriptor2);

		// �����_�}�b�`���O
		vector<DMatch> allMatch, goodMatch;
		BFMatcher matcher(NORM_HAMMING);
		matcher.match(descriptor1, descriptor2, allMatch);

		// ���Ă�������_�̂݃s�b�N�A�b�v
		for (int i = 0; i < (int)allMatch.size(); i++) {
			if (allMatch[i].distance < featureLEVEL) goodMatch.push_back(allMatch[i]);
		}

		// �`��
		Mat V_dst;
		drawMatches(src1, keypoint1, src2, keypoint2, goodMatch, V_dst);
		//if (h == 0) { V_dst.copyTo(img_SRC); }	// �m�F�p

		// ��v��
		float matchRate = (float)goodMatch.size() / (float)keypoint1.size();
		//cout << "��v��: " << matchRate << endl;	// �m�F�p
		cout << "keypoint1: " << (int)keypoint1.size() << endl;
		cout << "keypoint2: " << (int)keypoint2.size() << endl;
		cout << "allMatch: " << (int)allMatch.size() << endl;
		cout << "goodMatch: " << (int)goodMatch.size() << endl;

		// �����_�Ή��t��
		v_shift.push_back(V_dst);

		int index1, index2, shift_aveX = 0, shift_aveY = 0, occ_point = 0;
		double s_aX, s_aY;
		for (int i = 0; i < (int)goodMatch.size(); i++) {
			index1 = keypoint1[goodMatch[i].queryIdx].pt.y * video_OCC.XSIZE + keypoint1[goodMatch[i].queryIdx].pt.x;
			index2 = keypoint2[goodMatch[i].trainIdx].pt.y * video_OCC.XSIZE + keypoint2[goodMatch[i].trainIdx].pt.x;
			if (video_OCC.Image[h].data[index1] != 0 || video_OCC.Image[h+1].data[index2] != 0) { occ_point++; }
			else {
				shift_aveX = shift_aveX + (keypoint2[goodMatch[i].trainIdx].pt.x - keypoint1[goodMatch[i].queryIdx].pt.x);
				shift_aveY = shift_aveY + (keypoint2[goodMatch[i].trainIdx].pt.y - keypoint1[goodMatch[i].queryIdx].pt.y);
				//cout << "   " << (int)(keypoint2[goodMatch[i].trainIdx].pt.x - keypoint1[goodMatch[i].queryIdx].pt.x) << " : " << (int)(keypoint2[goodMatch[i].trainIdx].pt.y - keypoint1[goodMatch[i].queryIdx].pt.y) << endl;	// �m�F�p
			}
		}
		s_aX = (double)shift_aveX / ((int)goodMatch.size());
		s_aY = (double)shift_aveY / ((int)goodMatch.size());
		shift_aveX = (int)round(s_aX), shift_aveY = (int)round(s_aY);
		V_shift.push_back(Point2i(shift_aveX, shift_aveY));
		cout << "shift_ave: " << Point2i(shift_aveX, shift_aveY) << endl;	// �m�F�p
		cout << "shift_ave(X): " << s_aX/* shift_aveX*/ << endl;
		cout << "shift_ave(Y): " << s_aY/*shift_aveY*/ << endl;
	}
	V_shift.push_back(Point2i(0, 0));
}
void VideoShift::changeLEVEL(int change_level) {
	int before_lebel = shift_level;
	shift_level = change_level;

	int change = 1;
	int distance = shift_level - before_lebel;
	if (distance > 0) {
		for (int i = 0; i < distance; i++) {
			change = change * 2;
			xsize = xsize / 2;
			ysize = ysize / 2;
		}
	}
	else if (distance < 0) { cout << "WARRNING: VideoShift's pyr_LEVEL can not change-up!" << endl; }
	else { cout << "WARRNING: VideoShift's pyr_LEVEL is not change!" << endl; }

	Point2i tempPoint, sum = Point2i(0, 0);
	for (int i = 0; i < zsize - 1; i++) {
		tempPoint = V_shift[i];
		if (i % change == 0) {
			sum = sum + tempPoint;
			//cout << sum << " , " << tempPoint << endl;	// �m�F�p
			V_shift[i].x = (int)(sum.x / change);
			V_shift[i].y = (int)(sum.y / change);
			sum = Point2i(0, 0);
		}
		else {
			V_shift[i] = Point2i(0, 0);
			sum = sum + tempPoint;
		}
		//cout << tempPoint << " -> " << V_shift[i] << endl;	// �m�F�p
	}
}
void VideoShift::clear() {
	v_shift.clear();
	V_shift.clear();
}

void Read();	// �t�@�C���ǂݍ���
void Out();		// �t�@�C�������o��

void InpaintingInitialisation_feature(VideoSet&, ShiftMap&, VideoShift&);	// Inpainting initialisation with VideoShift
void PatchMatching(VideoSet&, ShiftMap&, VideoShift&, vector<Point3i>&, vector<Point3i>&);	// Patch Match Algorithm in initialisation
void PatchMatching_ANN(VideoSet&, ShiftMap&, vector<Point3i>&, Video&);
void PatchMatching_ANN(VideoSet&, ShiftMap&, vector<Point3i>&);				// Patch Match Algorithm (ANN)with VideoShift
void Reconstruction_first(VideoSet&, ShiftMap&, vector<Point3i>&, Video&);	// Reconstruction of U&T at first
void Reconstruction(VideoSet&, ShiftMap&, vector<Point3i>&);					// Reconstruction of U&T

int unoccluded_checker(Point3i center, Video& occluded) {
	int number_occ = 0;
	for (int z_p = center.z + PATCHstart; z_p < center.z + PATCHend; z_p++) {
		for (int y_p = center.y + PATCHstart; y_p < center.y + PATCHend; y_p++) {
			for (int x_p = center.x + PATCHstart; x_p < center.x + PATCHend; x_p++) {
				if (x_p >= 0 && x_p < occluded.XSIZE && y_p >= 0 && y_p < occluded.YSIZE && z_p >= 0 && z_p < occluded.ZSIZE) {
					if (occluded.Image[z_p].data[y_p * occluded.XSIZE + x_p] != 0) {
						number_occ++;
					}
				}
			}
		}
	}
	return number_occ;
}

double SHIGMA(vector<double>&);							// sort & return 75th percentile
void firstReconstructionU(VideoSet&, ShiftMap&);		// Reconstruction of U at first
void firstReconstruction(Video, VideoSet&, int);		// Reconstruction at first
//void annealingReconstruction(VideoSet&, ShiftMap&);

void Gamma_OCC_MRF_GaussSeidel_Color(VideoPyramid&);	// �����I�m�C�Y����(�s���~�b�h���)
void Gamma_OCC_MRF_GaussSeidel_Color(VideoPyramid&, vector<Video>&);
void MSE(Video&, Video&);