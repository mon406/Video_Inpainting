#ifndef __INCLUDED_H_Make_Blind_Video__
#define __INCLUDED_H_Make_Blind_Video__

#include "main.h"

/* --- �p�m���}�摜����r�f�I�쐬 ---------------------
	src.jpg			���͉摜
	src.avi			�o�͓��摜
�i�p�m���}�摜����r�f�I���쐬����j
 ------------------------------------------------------ */
void Read_Make_Video() {
	string file_src_img = "img\\panorama.jpg";	// ���͉摜�̃t�@�C����
	string file_dst_video = "video\\src.avi";	// �o�͓��摜�̃t�@�C����

	img_src = imread(file_src_img, 1);		// ���͉摜�i�J���[�j�̓ǂݍ���
	width = 256;				// ���摜�̏c��
	height = 128;
	fps = 30;					// �t���[�����[�g���擾

	int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');	// AVI�`�����w��
	// * �G���R�[�h�`�� "XVID" = AVI, "MP4V" = MPEG4, "WMV1" = WMV

	// �������ʂ̕ۑ�(��C���摜)
	VideoWriter recode(file_dst_video, fourcc, fps, Size(width, height), true);
	frame = 0;	// �t���[�����J�E���g
	int x_pix = 450;
	int y_pix = 550;
	while (1) {
		if (frame + width >= img_src.cols) { cout << "ERROR : frame number is over!" << endl; break; }
		if (frame >= 180) { break; }

		// ���͉摜�i�J���[�j�̉摜����
		Rect rect = Rect(x_pix, y_pix, width, height);
		Mat cut_image(img_src, rect);
		//x_pix++;							// A(���ɂP�s�N�Z�����ړ�)
		//if (x_pix + width >= img_src.cols) { cout << "ERROR : frame pixle is over!" << endl; break; }
		if (frame < 50) { x_pix++; }		// B(�����`��Ɉړ�)
		else if (frame < 90) { y_pix++; }
		else if (frame < 140) { x_pix--; }
		else { y_pix--; }
		if (x_pix + width >= img_src.cols || x_pix < 0 || y_pix + height >= img_src.rows || y_pix < 0) {
			cout << "ERROR : frame pixle is over!" << endl;
			break;
		}
		//if (frame < 70) { x_pix++; }		// C(Z�`�Ɉړ�)
		//else if (frame < 110) { 
		//	x_pix--;
		//	y_pix++;
		//}
		//else { x_pix++; }
		//if (x_pix + width >= img_src.cols || x_pix < 0 || y_pix + height >= img_src.rows || y_pix < 0) {
		//	cout << "ERROR : frame pixle is over!" << endl;
		//	break;
		//}

		cut_image.copyTo(video[frame]);	// ���摜����������

		// �o�͓��摜����������
		video[frame].copyTo(img_dst);
		recode << img_dst;
		frame++;

		imshow(win_dst, img_dst);	// �o�͓��摜��\��

		waitKey(33);	// 1000ms/30fps=33ms�҂�
	}

	/* �m�F */
	cout << "Size: ( " << width << " : " << height << " )" << endl;
	cout << "fps: " << fps << endl;
	cout << "frame: " << frame << endl;
}

#endif