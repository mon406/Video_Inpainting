#ifndef __INCLUDED_H_Make_Blind_Video__
#define __INCLUDED_H_Make_Blind_Video__

#include "main.h"

/* --- パノラマ画像からビデオ作成 ---------------------
	src.jpg			入力画像
	src.avi			出力動画像
（パノラマ画像からビデオを作成する）
 ------------------------------------------------------ */
void Read_Make_Video() {
	string file_src_img = "img\\panorama.jpg";	// 入力画像のファイル名
	string file_dst_video = "video\\src.avi";	// 出力動画像のファイル名

	img_src = imread(file_src_img, 1);		// 入力画像（カラー）の読み込み
	width = 256;				// 動画像の縦横
	height = 128;
	fps = 30;					// フレームレートを取得

	int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');	// AVI形式を指定
	// * エンコード形式 "XVID" = AVI, "MP4V" = MPEG4, "WMV1" = WMV

	// 処理結果の保存(補修動画像)
	VideoWriter recode(file_dst_video, fourcc, fps, Size(width, height), true);
	frame = 0;	// フレーム数カウント
	int x_pix = 450;
	int y_pix = 550;
	while (1) {
		if (frame + width >= img_src.cols) { cout << "ERROR : frame number is over!" << endl; break; }
		if (frame >= 180) { break; }

		// 入力画像（カラー）の画像処理
		Rect rect = Rect(x_pix, y_pix, width, height);
		Mat cut_image(img_src, rect);
		//x_pix++;							// A(横に１ピクセル毎移動)
		//if (x_pix + width >= img_src.cols) { cout << "ERROR : frame pixle is over!" << endl; break; }
		if (frame < 50) { x_pix++; }		// B(長方形上に移動)
		else if (frame < 90) { y_pix++; }
		else if (frame < 140) { x_pix--; }
		else { y_pix--; }
		if (x_pix + width >= img_src.cols || x_pix < 0 || y_pix + height >= img_src.rows || y_pix < 0) {
			cout << "ERROR : frame pixle is over!" << endl;
			break;
		}
		//if (frame < 70) { x_pix++; }		// C(Z形に移動)
		//else if (frame < 110) { 
		//	x_pix--;
		//	y_pix++;
		//}
		//else { x_pix++; }
		//if (x_pix + width >= img_src.cols || x_pix < 0 || y_pix + height >= img_src.rows || y_pix < 0) {
		//	cout << "ERROR : frame pixle is over!" << endl;
		//	break;
		//}

		cut_image.copyTo(video[frame]);	// 動画像を書き込む

		// 出力動画像を書き込む
		video[frame].copyTo(img_dst);
		recode << img_dst;
		frame++;

		imshow(win_dst, img_dst);	// 出力動画像を表示

		waitKey(33);	// 1000ms/30fps=33ms待つ
	}

	/* 確認 */
	cout << "Size: ( " << width << " : " << height << " )" << endl;
	cout << "fps: " << fps << endl;
	cout << "frame: " << frame << endl;
}

#endif