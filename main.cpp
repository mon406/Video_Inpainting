#include "main.h"
#include <time.h>

int main()
{
	Read();
	clock_t start = clock();	// Clock timer start
	// ここに核となる処理を記述する
	//---------------------------------------------------------------------
	Video video_original = Video(width, height, 3);
	for (int t = 0; t < frame; t++) {
		video_original.push_back(video[t]);
	}
	VideoPyramid video_pyramid = VideoPyramid(video_original);	// calculate O,U,T pyramid
	VideoSet videoL;
	video_pyramid.output(L, videoL);
	ShiftMap SM = ShiftMap(videoL);					// shift map decide randomly
	
	// 動画の画像シフト情報取得
	VideoShift video_shift = VideoShift(video_original, 0, 30);
	video_shift.changeLEVEL(L);
	//VideoShift video_shift = VideoShift(videoL.Color, videoL.Occlusion, L, 5);

	// Initialisation
	InpaintingInitialisation_feature(videoL, SM, video_shift);
	video_pyramid.input(L, videoL);

	vector<Point3i> VS_P, nonVS_P;
	vector<Video> nonVS_OCC;	// MRFに使うVideoShift無しのオクルージョン
	Video nonVS_temp;
	Mat BlackImage, tempMat;
	for (int pLEVEL = 0; pLEVEL <= L; pLEVEL++) {
		BlackImage = Mat::zeros(video_pyramid.nLEVEL_Y_SIZE[pLEVEL], video_pyramid.nLEVEL_X_SIZE[pLEVEL], CV_8UC1);
		for (int time = 0; time < frame; time++) {
			tempMat = BlackImage.clone();
			nonVS_temp.push_back(tempMat);
		}
		nonVS_OCC.push_back(nonVS_temp);
		cout << " Finidhed copy Video (nonVS) at: " << pLEVEL << endl;	// 確認用
		//cout << "  Size : " << video_pyramid.nLEVEL_Y_SIZE[pLEVEL] << "×" << video_pyramid.nLEVEL_X_SIZE[pLEVEL] << endl;
		nonVS_temp = Video();
	}
	// iteration
	Video UP_img;
	for (int pLEVEL = L; pLEVEL >= 0; pLEVEL--) {
		int K = 0;
		double e = 1.0;
		int c_num = 0, avg_num;
		while (e > 0.1 && K < 20) {
			// UpSample
			if (pLEVEL < L && K == 0) {
				VideoCopy(videoL.Color, UP_img);
				video_pyramid.input(pLEVEL + 1, videoL);
				videoL = VideoSet();
				video_pyramid.output(pLEVEL, videoL);
				SM.upsample(videoL.Occlusion);

				video_shift.clear();
				video_shift = VideoShift(video_original, 0, 30);
				if(pLEVEL != 0){ video_shift.changeLEVEL(pLEVEL); }

				// Reconstruction at first
				//firstReconstructionU(videoL, SM);			// shiftmap upsampling
				firstReconstruction(UP_img, videoL, pLEVEL);	// color upsampling
				// 実行結果確認
				cout << "Finidhed upsampling & first reconstruction at: " << pLEVEL << endl;
			}

			// Patch Matching & Reconstruction with VideoShift
			if (K == 0) {
				VS_P.clear(), nonVS_P.clear();
				cout << "Matching & Reconstruction with VideoShift..." << endl;	// 実行確認
				PatchMatching(videoL, SM, video_shift, nonVS_P, VS_P);
				cout << "  VS_num: " << VS_P.size() << endl;			// 確認用
				cout << "  nonVS_num: " << nonVS_P.size() << endl;
				/*for (int b = 0; b < nonVS_P.size(); b++) {
					cout << "  　　　" << nonVS_P[b] << endl;	// 確認用
				}*/

				uchar gray = 255;
				for (int ind_noVS = 0; ind_noVS < nonVS_P.size(); ind_noVS++) {
					Point3i ind_pix = nonVS_P[ind_noVS];
					int index_label = ind_pix.y * videoL.X_SIZE + ind_pix.x;
					if (videoL.Occlusion.Image[ind_pix.z].data[index_label] != 0) {
						nonVS_OCC[pLEVEL].Image[ind_pix.z].data[index_label] = gray;
					}
				}
			}

			if (pLEVEL != -1) {
				Video Before;
				Video After;
				VideoCopyTo(videoL.Color, Before);
				if (K == 0) {
					for (int Z = 0; Z < videoL.Z_SIZE; Z++) {
						for (int Y = 0; Y < videoL.Y_SIZE; Y++) {
							for (int X = 0; X < videoL.X_SIZE; X++) {
								if (videoL.Occlusion.Image[Z].data[Y * videoL.X_SIZE + X] != 0) {
									c_num++;
								}
							}
						}
					}
				}

				cout << "PatchMaching..." << endl;		// 実行確認
				cout << "   " << nonVS_P.size() << endl;
				PatchMatching_ANN(videoL, SM, nonVS_P);	// Patch Match(ANN) with noVideoShift

				cout << "Reconstruction..." << endl;	// 実行確認
				Reconstruction(videoL, SM, nonVS_P);	// Reconstruction U&T

				VideoCopy(videoL.Color, After);
				double Unorm = 0;
				for (int t = 0; t < frame; t++) {
					Unorm = Unorm + (double)norm(Before.Image[t], After.Image[t], NORM_L2);
					//cout << (double)Unorm << endl;	// 確認用
				}
				avg_num = 3 * c_num;
				e = (double)(Unorm) / (double)(avg_num);
				cout << " e= " << (double)Unorm << " / " << (int)avg_num << endl;	// 確認用
				//if (pLEVEL == L && K == 0) { e = 0.1; }	// K(0~)回目で中断
				K++;

				Before = Video();
				After = Video();
				// 実行結果確認
				cout << "成功: " << pLEVEL << ":" << K << ":" << e << endl;
			}
			else {
				cout << "First: " << pLEVEL << endl;
				e = 0.1;
			}
		}
	}
	video_pyramid.input(0, videoL);

	/* 出力画像確認 */
	for (int i = 0; i < frame; i++) {
		video_pyramid.nLEVEL_video[0].Color.Image[i].copyTo(video[i]);
	}
	//videoL.Color.Image[0].copyTo(img_SRC);
	//video_pyramid.nLEVEL_video[0].Color.Image[0].copyTo(img_SRC);
	nonVS_OCC[0].Image[10].copyTo(img_SRC);

	/*width = videoL.X_SIZE;
	height = videoL.Y_SIZE;
	cout << width << ":" << height << ":" << frame << endl;
	for (int i = 0; i < frame; i++) { videoL.Color.Image[i].copyTo(video[i]); }*/

	/* MRF画像処理 */
	cout << "Gamma_OCC_MRF_GaussSeidel..." << endl;	// 実行確認
	Gamma_OCC_MRF_GaussSeidel_Color(video_pyramid, nonVS_OCC);	// img_dst2に出力
	//Gamma_OCC_MRF_GaussSeidel_Color(video_pyramid);
	/*for(int i = 0; i < frame; i++) {
		img_dst2[i].copyTo(video[i]);
	}*/
	//---------------------------------------------------------------------
	clock_t end = clock();	// Clock timer end
	const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
	cout << "time " << time << "[ms]" << endl;

	MSE(video_pyramid.nLEVEL_video[0].Color, video_pyramid.nLEVEL_video[0].Occlusion);
	for (int i = 0; i < frame; i++) {
		img_dst2[i].copyTo(orig_video[i]);
	}

	Out();
	return 0;
}

// ファイル読み込み
void Read() {
	string file_src = "C:\\Users\\mon25\\Desktop\\video_inpainting\\src.avi";			// 入力動画像のファイル名
	string file_src2 = "C:\\Users\\mon25\\Desktop\\video_inpainting\\occlusion.png";	// 補修領域画像のファイル名
	
	img_mask = imread(file_src2, 0);	// マスク画像（グレースケール）の読み込み
	threshold(img_mask, img_mask, 100, 255, THRESH_BINARY);	// マスク画像の2値変換
	Mat img_mask_not;						// マスク画像（白黒反転）
	bitwise_not(img_mask, img_mask_not);

	// ビデオファイルを開く
	VideoCapture capture(file_src);
	if (!capture.isOpened()) {
		cout << "ビデオファイルが開けません。" << endl;
	}
	width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);		// フレーム横幅を取得
	height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);	// フレーム縦幅を取得
	fps = capture.get(cv::CAP_PROP_FPS);					// フレームレートを取得

	frame = 0;	// フレーム数カウント
	while (1) {
		capture >> img_src;								// 入力動画像（カラー）の読み込み
		frame++;
		if (img_src.empty()==true) { break; }

		// 画像処理
		//img_src.copyTo(img_SRC, img_mask_not);			// 補修する動画像作成（カラー）
		img_src.copyTo(img_SRC);
		//cvtColor(img_src, img_src2, COLOR_RGB2GRAY);	// 入力動画像（グレースケール）の読み込み
		//img_src2.copyTo(img_SRC2, img_mask_not);		// 補修する動画像作成（グレースケール）

		img_src.copyTo(orig_video[frame - 1]);
		img_src.copyTo(video[frame - 1], img_mask_not);	// 動画像を書き込む

		waitKey(33);	// 1000ms/30fps=33ms待つ
	}
	frame = frame - 1;	// 動画終了点
	/* 確認 */
	cout << "Size: ( " << width << " : " << height << " )" << endl;
	cout << "fps: " << fps << endl;
	cout << "frame: " << frame << endl;
}
// ファイル書き出し
void Out() {
	string file_dst = "C:\\Users\\mon25\\Desktop\\video_inpainting\\dst.avi";	// 出力動画像のファイル名
	string file_dst2 = "C:\\Users\\mon25\\Desktop\\video_inpainting\\dst2.jpg";
	string file_dst3 = "C:\\Users\\mon25\\Desktop\\video_inpainting\\dst3.avi";

	// ウィンドウ生成
	//namedWindow(win_src, WINDOW_AUTOSIZE);
	//namedWindow(win_dst, WINDOW_AUTOSIZE);

	int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');	// AVI形式を指定
	// * エンコード形式 "XVID" = AVI, "MP4V" = MPEG4, "WMV1" = WMV

	// 処理結果の保存(途中画像確認用)
	imwrite(file_dst2, img_SRC);
	// 処理結果の保存(補修動画像)
	VideoWriter recode(file_dst, fourcc, fps, Size(width, height),true);
	VideoWriter recode2(file_dst3, fourcc, fps, Size(width, height), true);
	int frame_counter = 0;
	while (1) {
		if (frame_counter >= frame) { break; }
		video[frame_counter].copyTo(img_dst);
		recode << img_dst;
		orig_video[frame_counter].copyTo(img_dst3);
		recode2 << img_dst3;
		frame_counter++;

		imshow(win_dst, img_dst);	// 出力動画像を表示
		
		waitKey(33);	// 1000ms/30fps=33ms
	}

	//waitKey(0); // キー入力待ち
}

// Inpainting initialisation with VideoShift
void InpaintingInitialisation_feature(VideoSet& imageL, ShiftMap& smL, VideoShift& Video_shift) {
	Video current_occlusion;			// current occlusion H'
	Video new_video, new_texture;

	VideoCopyTo(imageL.Occlusion, current_occlusion);
	VideoCopy(imageL.Color, new_video);
	VideoCopy(imageL.Texture, new_texture);

	// Patch Match & Reconstruction with VideoShift
	vector<Point3i> VideoS, nonVideoS;
	PatchMatching(imageL, smL, Video_shift, nonVideoS, VideoS);
	int VS_num = VideoS.size();
	int nonVS_num = nonVideoS.size();
	cout << "  VS_num: " << VS_num << endl;			// 確認用
	cout << "  nonVS_num: " << nonVS_num << endl;
	// Erosion(H',B)
	Point3i Point;
	uchar gray = (uchar)0;
	for (int occl_num = 0; occl_num < VS_num; occl_num++) {
		Point = VideoS[occl_num];
		current_occlusion.Image[Point.z].data[Point.y * current_occlusion.XSIZE + Point.x] = gray;
	}
	gray = (uchar)255;
	for (int occl_num = 0; occl_num < nonVS_num; occl_num++) {
		Point = nonVideoS[occl_num];
		current_occlusion.Image[Point.z].data[Point.y * current_occlusion.XSIZE + Point.x] = gray;
	}

	Point3i pix, A, B;
	int indexP;
	while (nonVS_num != 0) {
		vector<Point3i> OccPixel;	// the position of current layer to inpaint aH'
		vector<Point3i> OccPixel2;
		for (int ind = 0; ind < nonVS_num; ind++) {
			// aH' <- H'
			pix = nonVideoS[ind];
			//cout << " pix : " << pix << endl;	// 確認用
			indexP = pix.y * current_occlusion.XSIZE + pix.x;
			//if (current_occlusion.Image[pix.z].data[indexP] != 0) {
				if (current_occlusion.Image[pix.z].data[indexP - 1] == 0 && (pix.x - 1) >= 0) {
					OccPixel.push_back(pix);
					//cout << "   left : " << pix << endl;
				}
				else if (current_occlusion.Image[pix.z].data[indexP + 1] == 0 && (pix.x + 1) < current_occlusion.XSIZE) {
					OccPixel.push_back(pix);
					//cout << "   right: " << pix << endl;
				}
				else if (current_occlusion.Image[pix.z].data[indexP - current_occlusion.XSIZE] == 0 && (pix.y - 1) >= 0) {
					OccPixel.push_back(pix);
					//cout << "   up   : " << pix << endl;
				}
				else if (current_occlusion.Image[pix.z].data[indexP + current_occlusion.XSIZE] == 0 && (pix.y + 1) < current_occlusion.YSIZE) {
					OccPixel.push_back(pix);
					//cout << "   down : " << pix << endl;
				}
				else {
					OccPixel2.push_back(pix);
					/*cout << "occlusion pixel in occlusion: " << pix << endl;*/
				}
			//}
		}
		// ANN PatchMaching
		PatchMatching_ANN(imageL, smL, OccPixel, current_occlusion);

		// Reconstruction U&T
		Reconstruction_first(imageL, smL, OccPixel, current_occlusion);
		
		// Erosion(H',B)
		gray = (uchar)0;
		for (int occl_num = 0; occl_num < OccPixel.size(); occl_num++) {
			Point = OccPixel[occl_num];
			current_occlusion.Image[Point.z].data[Point.y * current_occlusion.XSIZE + Point.x] = gray;
		}
		OccPixel.clear();

		nonVS_num = OccPixel2.size();		// the number of current occlusion H'
		nonVideoS = OccPixel2;
		cout << "  nonVS_num: " << nonVS_num << endl;	// 確認用
		OccPixel2.clear();
	}
	VideoCopy(new_video, imageL.Color);
	VideoCopy(new_texture, imageL.Texture);
}

// Reconstruction of U at first by ShiftMap
void firstReconstructionU(VideoSet& Img, ShiftMap& Sm) {
	Vec3b color;
	Point3i ANNpoint;
	Point2i currentPoint;
	for (int z = 0; z < frame; z++) {
		for (int y = 0; y < Img.Y_SIZE; y++) {
			for (int x = 0; x < Img.X_SIZE; x++) {
				if (Img.Occlusion.Image[z].data[y * Img.X_SIZE + x] != 0) {
					currentPoint = Point2i(x, y);
					ANNpoint = Point3i(x + Sm.nnX(currentPoint, z), y + Sm.nnY(currentPoint, z), z + Sm.nnT(currentPoint, z));
					color = Img.Color.Image[ANNpoint.z].at<Vec3b>(ANNpoint.y, ANNpoint.x);
					Img.Color.Image[z].at<Vec3b>(y, x) = color;
				}
			}
		}
	}
}
// Reconstruction at first
void firstReconstruction(Video UP_img, VideoSet& Img, int pyrLEVEL)
{
	Video UPimg = Video(UP_img.XSIZE*2, UP_img.YSIZE * 2, 3);
	Mat UPtemp;
	if (pyrLEVEL == L) { cout << " RROR: Upsample without L!" << endl; }
	for (int z = 0; z < frame; z++) {
		pyrUp(UP_img.Image[z], UPtemp);
		UPimg.push_back(UPtemp);
	}

	Vec3b color;
	Point2i currentPoint;
	for (int z = 0; z < frame; z++) {
		for (int y = 0; y < Img.Y_SIZE; y++) {
			for (int x = 0; x < Img.X_SIZE; x++) {
				if (Img.Occlusion.Image[z].data[y * Img.X_SIZE + x] != 0) {
					currentPoint = Point2i(x, y);
					color = UPimg.Image[z].at<Vec3b>(currentPoint.y, currentPoint.x);
					Img.Color.Image[z].at<Vec3b>(y, x) = color;
				}
			}
		}
	}
}

// Patch Match & Reconstruction with VideoShift
void PatchMatching(VideoSet& image, ShiftMap& sm, VideoShift& vs, vector<Point3i>& outNoVideoShift, vector<Point3i>& outVideoShift) {
	int noVS_number = 0;
	vector<Point3i> noVS;	// VideoShift先のないオクルージョン
	int occ_checker, V_Stemp;
	int back, front;
	Point3i P, Point, minCost_Point;
	Vec3b color;
	for (int ZP = 0; ZP < frame; ZP++) {
		for (int YP = 0; YP < image.Y_SIZE; YP++) {
			for (int XP = 0; XP < image.X_SIZE; XP++) {
				P = Point3i(XP, YP, ZP);
				occ_checker = unoccluded_checker(P, image.Occlusion);
				if (occ_checker != 0) {				// occlusion H~ (dilated H)
					// VideoShift先のシフトマップ取得
					Point = P;
					V_Stemp = 200;	// 200 > frame が条件
					for (int VS_p = ZP; VS_p < frame; VS_p++) {
						if (V_Stemp == 200) {
							Point.x = Point.x + vs.V_shift[VS_p].x;
							Point.y = Point.y + vs.V_shift[VS_p].y;
							Point.z++;

							if (Point.z >= image.Z_SIZE - 1) { break; }
							if (Point.x >= 0 && Point.x < image.X_SIZE && Point.y >= 0 && Point.y < image.Y_SIZE) {
								occ_checker = unoccluded_checker(Point, image.Occlusion);
								if (occ_checker == 0) {
									minCost_Point = Point;
									V_Stemp = Point.z;
								}
							}
						}
						else { break; }
					}
					Point = P;
					for (int VS_p = ZP; VS_p > 0; VS_p--) {
						if (Point.z <= 0) { break; }
						Point.x = Point.x - vs.V_shift[VS_p - 1].x;
						Point.y = Point.y - vs.V_shift[VS_p - 1].y;
						Point.z--;

						if (Point.x >= 0 && Point.x < image.X_SIZE && Point.y >= 0 && Point.y < image.Y_SIZE) {
							occ_checker = unoccluded_checker(Point, image.Occlusion);
							if (occ_checker == 0) {
								back = abs(V_Stemp - ZP);
								front = abs(ZP - Point.z);
								if (V_Stemp == 200) {
									minCost_Point = Point;
									V_Stemp = Point.z;
								}
								else if (front < back) {
									minCost_Point = Point;
									V_Stemp = Point.z;
								}
							}
						}
					}

					if (V_Stemp != 200) {
						//cout << P << " -> " << minCost_Point << endl;	// 確認用
						minCost_Point = minCost_Point - P;
						sm.put(P, minCost_Point);	// VideoShift先のシフトマップ更新

						if (image.Occlusion.Image[ZP].data[YP * image.X_SIZE + XP] != 0) {
							outVideoShift.push_back(P);
							minCost_Point = minCost_Point + P;

							// Reconstruction U&T
							Vec3b color;
							uchar tx, ty;
							color = image.Color.Image[minCost_Point.z].at<Vec3b>(minCost_Point.y, minCost_Point.x);	// ピクセル値（カラー）を取得
							image.Color.Image[P.z].at<Vec3b>(P.y, P.x) = color;
							int t_ind = minCost_Point.y * image.X_SIZE * 2 + minCost_Point.x;	// テクスチャを取得
							tx = image.Texture.Image[minCost_Point.z].data[t_ind];
							ty = image.Texture.Image[minCost_Point.z].data[t_ind + 1];
							t_ind = P.y * image.X_SIZE * 2 + P.x;
							image.Texture.Image[P.z].data[t_ind] = tx;
							image.Texture.Image[P.z].data[t_ind + 1] = ty;
						}
					}
					else {
						noVS.push_back(P);
						noVS_number++;
					}
				}
			}
		}
	}

	// 参照できなかった領域を保存
	//cout << "### no VideoShift pixel number is " << noVS_number << endl;	// 確認用
	for (int kk = 0; kk < noVS_number; kk++) {
		outNoVideoShift.push_back(noVS[kk]);
	}
	noVS.clear();
}
// Patch Match Algorithm (ANN探索)
void PatchMatching_ANN(VideoSet& image, ShiftMap& sm, vector<Point3i>& NoVideoShift, Video& NOW_OCC) {
	r_max = /*16*/frame;		// ※the maximum dimention of video
	double Round;			// シフトマップ周辺探索領域

	Point3i P, P3, A, B, C;	// 比較ピクセル
	Point2i P2, now2;
	double p, q;			// 比較コスト
	Point3i point_now, Point, minCost_Point;
	vector<Point3i> Point_abc;
	Patch PATCH;
	int occ_checker;
	//iteration
	for (int kk = 0; kk < 10; kk++) {
		Round = (int)(r_max * pow(Ro, kk) * RandUniform);
		//cout << " finish ANN : kk = " << (int)kk << " , Round = " << Round << endl;	// 確認用
		//cout << "       occlusion = " << (int)NoVideoShift.size() << endl;
		// 左、上、前のシフトマップ
		for (int pp = 0; pp < NoVideoShift.size(); pp++) {
			P = NoVideoShift[pp];			// Point_abc = a, b, c
			Point_abc.push_back(P);
			P2 = Point2i(P.x, P.y);
			A = Point3i(P.x - 1, P.y, P.z);
			Point_abc.push_back(A);
			B = Point3i(P.x, P.y - 1, P.z);
			Point_abc.push_back(B);
			C = Point3i(P.x, P.y, P.z - 1);
			Point_abc.push_back(C);
			// 最小コスト探索 in a,b,c
			p = DBL_MAX;	// 最大値を代入
			minCost_Point = P + sm.nn(P2, P.z);
			for (int Point_Index = 0; Point_Index < 4; Point_Index++) {
				point_now = Point_abc[Point_Index];
				now2 = Point2i(point_now.x, point_now.y);
				if (point_now.x >= 0 && point_now.x < image.X_SIZE && point_now.y >= 0 && point_now.y < image.Y_SIZE && point_now.z >= 0 && point_now.z < image.Z_SIZE) {
					Point = P + sm.nn(now2, point_now.z);
					if (Point.x > 0 && Point.x < image.X_SIZE && Point.y > 0 && Point.y < image.Y_SIZE && Point.z > 0 && Point.z < image.Z_SIZE) {
						occ_checker = unoccluded_checker(Point, image.Occlusion);
						if (occ_checker == 0) {
							PATCH = Patch(P, Point, image.Color, image.Texture, NOW_OCC, sm);
							q = PATCH.costfunction(1);
							if (p > q && q != 0) {
								p = q;
								minCost_Point = Point;
							}
							PATCH = Patch();
						}
					}
				}
			}
			minCost_Point = minCost_Point - P;
			sm.put(P, minCost_Point);
			Point_abc.clear();
		}

		// 右、下、後のシフトマップ
		for (int pp = 0; pp < NoVideoShift.size(); pp++) {
			P = NoVideoShift[pp];			// Point_abc = a, b, c
			Point_abc.push_back(P);
			P2 = Point2i(P.x, P.y);
			A = Point3i(P.x + 1, P.y, P.z);
			Point_abc.push_back(A);
			B = Point3i(P.x, P.y + 1, P.z);
			Point_abc.push_back(B);
			C = Point3i(P.x, P.y, P.z + 1);
			Point_abc.push_back(C);
			// 最小コスト探索 in a,b,c
			p = DBL_MAX;	// 最大値を代入
			minCost_Point = P + sm.nn(P2, P.z);
			for (int Point_Index = 0; Point_Index < 4; Point_Index++) {
				point_now = Point_abc[Point_Index];
				now2 = Point2i(point_now.x, point_now.y);
				if (point_now.x >= 0 && point_now.x < image.X_SIZE && point_now.y >= 0 && point_now.y < image.Y_SIZE && point_now.z >= 0 && point_now.z < image.Z_SIZE) {
					Point = P + sm.nn(now2, point_now.z);
					if (Point.x > 0 && Point.x < image.X_SIZE && Point.y > 0 && Point.y < image.Y_SIZE && Point.z > 0 && Point.z < image.Z_SIZE) {
						occ_checker = unoccluded_checker(Point, image.Occlusion);
						if (occ_checker == 0) {
							PATCH = Patch(P, Point, image.Color, image.Texture, NOW_OCC, sm);
							q = PATCH.costfunction(1);
							if (p > q && q != 0) {
								p = q;
								minCost_Point = Point;
							}
							PATCH = Patch();
						}
					}
				}
			}
			minCost_Point = minCost_Point - P;
			sm.put(P, minCost_Point);
			Point_abc.clear();
		}

		// シフトマップ先の周辺
		for (int pp = 0; pp < NoVideoShift.size(); pp++) {
			P = NoVideoShift[pp];
			P2 = Point2i(P.x, P.y);
			P3 = P + sm.nn(P2, P.z);
			//cout << "  around SM : " << P << " (SM: "<< P3 << " )" << endl;	// 確認用
			if (Round > 0) {
				// Point_abc:検索対象ピクセル
				int MaxRound = Round * 2 + 1;
				int rand_check = 1, kkk = 0;
				while (rand_check == 1 && kkk < 10) {
					point_now.x = (int)(rand() % MaxRound + (P3.x - Round));	// 比較先(ramdom)
					point_now.y = (int)(rand() % MaxRound + (P3.y - Round));
					point_now.z = (int)(rand() % MaxRound + (P3.z - Round));
					if (point_now.x > 0 && point_now.x < image.X_SIZE && point_now.y > 0 && point_now.y < image.Y_SIZE && point_now.z > 0 && point_now.z < image.Z_SIZE) {
						rand_check = 0;
					}
					kkk++;
				}
				if (rand_check == 1 && kkk == 10) { break; }
				/*for (int ZP = P3.z - Round; ZP <= P3.z + Round; ZP++) {
					for (int YP = P3.y - Round; YP <= P3.y + Round; YP++) {
						for (int XP = P3.x - Round; XP <= P3.x + Round; XP++) {
							Point = Point3i(XP, YP, ZP);
							if (XP > 0 && XP < image.X_SIZE && YP > 0 && YP < image.Y_SIZE && ZP > 0 && ZP < image.Z_SIZE) {
								Point_abc.push_back(Point);
							}
						}
					}
				}*/
				// 最小コスト探索 in Point_abc
				p = DBL_MAX;	// 最大値を代入
				minCost_Point = P3;
				/*for (int Point_Index = 0; Point_Index < Point_abc.size(); Point_Index++) {
					point_now = Point_abc[Point_Index];
					now2 = Point2i(point_now.x, point_now.y);
					Point = P + sm.nn(now2, point_now.z);
					if (Point.x > 0 && Point.x < image.X_SIZE && Point.y > 0 && Point.y < image.Y_SIZE && Point.z > 0 && Point.z < image.Z_SIZE) {
						occ_checker = unoccluded_checker(Point, image.Occlusion);
						if (occ_checker == 0) {
							PATCH = Patch(P, Point, image.Color, image.Texture, NOW_OCC, sm);
							q = PATCH.costfunction(1);
							if (p > q && q != 0) {
								p = q;
								minCost_Point = Point;
							}
							PATCH = Patch();
						}
					}
				}*/
				now2 = Point2i(point_now.x, point_now.y);
				Point = point_now + sm.nn(now2, point_now.z);
				if (Point.x > 0 && Point.x < image.X_SIZE && Point.y > 0 && Point.y < image.Y_SIZE && Point.z > 0 && Point.z < image.Z_SIZE) {
					occ_checker = unoccluded_checker(Point, image.Occlusion);
					if (occ_checker == 0) {
						PATCH = Patch(P, Point, image.Color, image.Texture, NOW_OCC, sm);
						q = PATCH.costfunction(1);
						if (p > q && q != 0) {
							//cout << "Point: " << P2 << " , cost: " << p << endl;	// 確認用
							p = q;
							minCost_Point = Point;
						}
						PATCH = Patch();
					}
				}
				minCost_Point = minCost_Point - P;
				sm.put(P, minCost_Point);
				Point_abc.clear();
			}
		}
	}

	///* シフトマップ確認 */
	//Point3i sm_check;
	//cout << "シフトマップ" << endl;
	//for (int i = 0; i < NoVideoShift.size(); i++) {
	//	Point = NoVideoShift[i];
	//	P2 = Point2i(Point.x, Point.y);
	//	sm_check = Point + sm.nn(P2, Point.z);
	//	cout << Point << " -> " << sm_check;
	//	if (image.Occlusion.Image[sm_check.z].data[sm_check.y * image.X_SIZE + sm_check.x] != 0) {
	//		cout << "   ###" << endl;
	//	}
	//	else { cout << endl; }
	//}
}
// Patch Match Algorithm (ANN探索) with VideoShift
void PatchMatching_ANN(VideoSet& image, ShiftMap& sm, vector<Point3i>& NoVideoShift) {
	r_max = /*4*/frame;		// ※the maximum dimention of video
	double Round;			// シフトマップ周辺探索領域

	Point3i P, P3, A, B, C;	// 比較ピクセル
	Point2i P2, now2;
	double p, q;			// 比較コスト
	Point3i point_now, Point, minCost_Point;
	vector<Point3i> Point_abc;
	Patch PATCH;
	int occ_checker;
	//iteration
	for (int kk = 0; kk < 3; kk++) {
		Round = (int)(r_max * pow(Ro, kk) * RandUniform);
		//cout << " finish ANN : kk = " << (int)kk << " , Round = " << Round << endl;	// 確認用
		// 左、上、前のシフトマップ
		for (int pp = 0; pp < NoVideoShift.size(); pp++) {
			P = NoVideoShift[pp];			// Point_abc = a, b, c
			Point_abc.push_back(P);
			P2 = Point2i(P.x, P.y);
			A = Point3i(P.x - 1, P.y, P.z);
			Point_abc.push_back(A);
			B = Point3i(P.x, P.y - 1, P.z);
			Point_abc.push_back(B);
			C = Point3i(P.x, P.y, P.z - 1);
			Point_abc.push_back(C);
			// 最小コスト探索 in a,b,c
			p = DBL_MAX;	// 最大値を代入
			minCost_Point = P + sm.nn(P2, P.z);
			for (int Point_Index = 0; Point_Index < 4; Point_Index++) {
				point_now = Point_abc[Point_Index];
				now2 = Point2i(point_now.x, point_now.y);
				if (point_now.x >= 0 && point_now.x < image.X_SIZE && point_now.y >= 0 && point_now.y < image.Y_SIZE && point_now.z >= 0 && point_now.z < image.Z_SIZE) {
					Point = P + sm.nn(now2, point_now.z);
					if (Point.x > 0 && Point.x < image.X_SIZE && Point.y > 0 && Point.y < image.Y_SIZE && Point.z > 0 && Point.z < image.Z_SIZE) {
						occ_checker = unoccluded_checker(Point, image.Occlusion);
						if (occ_checker == 0) {
							PATCH = Patch(P, Point, image.Color, image.Texture, image.Occlusion, sm);
							q = PATCH.costfunction(1);
							if (p > q && q != 0) {
								p = q;
								minCost_Point = Point;
							}
							PATCH = Patch();
						}
					}
				}
			}
			minCost_Point = minCost_Point - P;
			sm.put(P, minCost_Point);
			Point_abc.clear();
		}

		// 右、下、後のシフトマップ
		for (int pp = 0; pp < NoVideoShift.size(); pp++) {
			P = NoVideoShift[pp];			// Point_abc = a, b, c
			Point_abc.push_back(P);
			P2 = Point2i(P.x, P.y);
			A = Point3i(P.x + 1, P.y, P.z);
			Point_abc.push_back(A);
			B = Point3i(P.x, P.y + 1, P.z);
			Point_abc.push_back(B);
			C = Point3i(P.x, P.y, P.z + 1);
			Point_abc.push_back(C);
			// 最小コスト探索 in a,b,c
			p = DBL_MAX;	// 最大値を代入
			minCost_Point = P + sm.nn(P2, P.z);
			for (int Point_Index = 0; Point_Index < 4; Point_Index++) {
				point_now = Point_abc[Point_Index];
				now2 = Point2i(point_now.x, point_now.y);
				if (point_now.x >= 0 && point_now.x < image.X_SIZE && point_now.y >= 0 && point_now.y < image.Y_SIZE && point_now.z >= 0 && point_now.z < image.Z_SIZE) {
					Point = P + sm.nn(now2, point_now.z);
					if (Point.x > 0 && Point.x < image.X_SIZE && Point.y > 0 && Point.y < image.Y_SIZE && Point.z > 0 && Point.z < image.Z_SIZE) {
						occ_checker = unoccluded_checker(Point, image.Occlusion);
						if (occ_checker == 0) {
							PATCH = Patch(P, Point, image.Color, image.Texture, image.Occlusion, sm);
							q = PATCH.costfunction(1);
							if (p > q && q != 0) {
								p = q;
								minCost_Point = Point;
							}
							PATCH = Patch();
						}
					}
				}
			}
			minCost_Point = minCost_Point - P;
			sm.put(P, minCost_Point);
			Point_abc.clear();
		}

		// シフトマップ先の周辺
		for (int pp = 0; pp < NoVideoShift.size(); pp++) {
			P = NoVideoShift[pp];
			P2 = Point2i(P.x, P.y);
			P3 = P + sm.nn(P2, P.z);
			//cout << "  around SM : " << P << " (SM: "<< P3 << " )" << endl;	// 確認用
			if (Round > 0) {
				// Point_abc:検索対象ピクセル
				int MaxRound = Round * 2 + 1;
				int rand_check = 1, kkk = 0;
				while (rand_check == 1 && kkk < 10) {
					point_now.x = (int)(rand() % MaxRound + (P3.x - Round));	// 比較先(ramdom)
					point_now.y = (int)(rand() % MaxRound + (P3.y - Round));
					point_now.z = (int)(rand() % MaxRound + (P3.z - Round));
					if (point_now.x > 0 && point_now.x < image.X_SIZE && point_now.y > 0 && point_now.y < image.Y_SIZE && point_now.z > 0 && point_now.z < image.Z_SIZE) {
						rand_check = 0;
					}
					kkk++;
				}
				if (rand_check == 1 && kkk == 10) { break; }
				/*for (int ZP = P3.z - Round; ZP <= P3.z + Round; ZP++) {
					for (int YP = P3.y - Round; YP <= P3.y + Round; YP++) {
						for (int XP = P3.x - Round; XP <= P3.x + Round; XP++) {
							Point = Point3i(XP, YP, ZP);
							if (XP > 0 && XP < image.X_SIZE && YP > 0 && YP < image.Y_SIZE && ZP > 0 && ZP < image.Z_SIZE) {
								Point_abc.push_back(Point);
							}
						}
					}
				}*/
				// 最小コスト探索 in Point_abc
				p = DBL_MAX;	// 最大値を代入
				minCost_Point = P3;
				/*for (int Point_Index = 0; Point_Index < Point_abc.size(); Point_Index++) {
					point_now = Point_abc[Point_Index];
					now2 = Point2i(point_now.x, point_now.y);
					Point = P + sm.nn(now2, point_now.z);
					if (Point.x > 0 && Point.x < image.X_SIZE && Point.y > 0 && Point.y < image.Y_SIZE && Point.z > 0 && Point.z < image.Z_SIZE) {
						occ_checker = unoccluded_checker(Point, image.Occlusion);
						if (occ_checker == 0) {
							PATCH = Patch(P, Point, image.Color, image.Texture, image.Occlusion, sm);
							q = PATCH.costfunction(1);
							if (p > q && q != 0) {
								p = q;
								minCost_Point = Point;
							}
							PATCH = Patch();
						}
					}
				}*/
				now2 = Point2i(point_now.x, point_now.y);
				Point = point_now + sm.nn(now2, point_now.z);
				if (Point.x > 0 && Point.x < image.X_SIZE && Point.y > 0 && Point.y < image.Y_SIZE && Point.z > 0 && Point.z < image.Z_SIZE) {
					occ_checker = unoccluded_checker(Point, image.Occlusion);
					if (occ_checker == 0) {
						PATCH = Patch(P, Point, image.Color, image.Texture, image.Occlusion, sm);
						q = PATCH.costfunction(3);
						if (p > q && q != 0) {
							//cout << "Point: " << P2 << " , cost: " << p << endl;	// 確認用
							p = q;
							minCost_Point = Point;
						}
						PATCH = Patch();
					}
				}
				minCost_Point = minCost_Point - P;
				sm.put(P, minCost_Point);
				Point_abc.clear();
			}
		}
	}

	///* シフトマップ確認 */
	//Point3i sm_check;
	//cout << "シフトマップ" << endl;
	//for (int i = 0; i < NoVideoShift.size(); i++) {
	//	Point = NoVideoShift[i];
	//	P2 = Point2i(Point.x, Point.y);
	//	sm_check = Point + sm.nn(P2, Point.z);
	//	cout << Point << " -> " << sm_check;
	//	if (image.Occlusion.Image[sm_check.z].data[sm_check.y * image.X_SIZE + sm_check.x] != 0) {
	//		cout << "   ###" << endl;
	//	}
	//	else { cout << endl; }
	//}
}

// Reconstruction of U&T at first
void Reconstruction_first(VideoSet& Img, ShiftMap& Sm, vector<Point3i>& inpaintP, Video& nowOCC) {
	Patch PATCH;
	double shigma, Spq;
	vector<double> Cost, sqrtCost;
	Point3i A, B, C;
	Point2i P2;
	double costSUM, costSUMsqrt;
	vector<double> colorR, colorG, colorB;
	double avgColorR, avgColorG, avgColorB;
	Vec3b color;
	uchar r, g, b;
	vector<float> TX, TY;
	float tx, ty;
	double avgX, avgY;
	int non, P_SIZEint;
	for (int i = 0; i < inpaintP.size(); i++) {
		A = inpaintP[i];
		non = 0;
		for (int z = A.z + PATCHstart; z < A.z + PATCHend; z++) {
			for (int y = A.y + PATCHstart; y < A.y + PATCHend; y++) {
				for (int x = A.x + PATCHstart; x < A.x + PATCHend; x++) {
					if (x >= 0 && x < Img.X_SIZE && y >= 0 && y < Img.Y_SIZE && z >= 0 && z < Img.Z_SIZE) {
						if (nowOCC.Image[z].data[y * Img.X_SIZE + x] == 0) {
							C = Point3i(x, y, z);
							P2 = Point2i(x, y);
							B = C + Sm.nn(P2, C.z);
							PATCH = Patch(C, B, Img.Color, Img.Texture, nowOCC, Sm);
							C = A + Sm.nn(P2, C.z);
							if (C.x >= 0 && C.x < Img.X_SIZE && C.y >= 0 && C.y < Img.Y_SIZE && C.z >= 0 && C.z < Img.Z_SIZE) {
								Cost.push_back(PATCH.costfunction(3));
								sqrtCost.push_back(PATCH.costfunction(2));

								color = Img.Color.Image[C.z].at<Vec3b>(C.y, C.x);	// ピクセル値（カラー）を取得
								r = color[2];	// R,G,B値に分解
								g = color[1];
								b = color[0];
								colorR.push_back((double)r);
								colorG.push_back((double)g);
								colorB.push_back((double)b);
								tx = (float)Img.Texture.Image[C.z].data[C.y * Img.X_SIZE * 2 + C.x * 2 + 0];	// テクスチャ特徴を取得
								ty = (float)Img.Texture.Image[C.z].data[C.y * Img.X_SIZE * 2 + C.x * 2 + 1];
								TX.push_back(tx);
								TY.push_back(ty);
							}
							else { non++; }
							PATCH = Patch();
						}
						else { non++; }
					}
					else { non++; }
				}
			}
		}
		//cout << "  non = " << non << "  (/125)" << endl;	// 確認用

		// sort Cost and return 75 persentile
		shigma = SHIGMA(sqrtCost);
		Spq = 0.0;
		P_SIZEint = PATCHSIZEint * PATCHSIZEint * PATCHSIZEint - non;
		for (int ii = 0; ii < P_SIZEint; ii++) {
			Spq = Spq + (double)exp(-Cost[ii] / (2 * shigma * shigma));
		}
		//cout << "Spq： " << Spq << endl;	// 確認用

		if (Spq > 0 && Spq < 10000) {	// ※-nan(ind),1e+06に対する対応
			// reconstruct u(p)
			avgColorR = 0.0, avgColorG = 0.0, avgColorB = 0.0;
			for (int ii = 0; ii < P_SIZEint; ii++) {
				avgColorR = avgColorR + Spq * colorR[ii];
				avgColorG = avgColorG + Spq * colorG[ii];
				avgColorB = avgColorB + Spq * colorB[ii];
			}
			avgColorR = avgColorR / P_SIZEint;
			avgColorG = avgColorG / P_SIZEint;
			avgColorB = avgColorB / P_SIZEint;
			r = (uchar)(avgColorR / Spq);	// R,G,B値を処理
			g = (uchar)(avgColorG / Spq);
			b = (uchar)(avgColorB / Spq);
			//color = Img.Color.Image[A.z].at<Vec3b>(A.y, A.x);	// 確認用
			//cout << color << " -> ";
			color = Vec3b(b, g, r);				// R,G,B値からピクセル値（カラー）を生成
			Img.Color.Image[A.z].at<Vec3b>(A.y, A.x) = color;	// ピクセル値（カラー）を設定
			//cout << color << endl;	// 確認用

			// reconstruct t(p)
			avgX = 0.0, avgY = 0.0;
			for (int ii = 0; ii < P_SIZEint; ii++) {
				avgX = avgX + Spq * TX[ii];
				avgY = avgY + Spq * TY[ii];
			}
			avgX = avgX / P_SIZEint;
			avgY = avgY / P_SIZEint;
			tx = (float)(avgX / Spq);
			ty = (float)(avgY / Spq);
			Img.Texture.Image[A.z].data[A.y * Img.X_SIZE * 2 + A.x * 2 + 0] = tx;
			Img.Texture.Image[A.z].data[A.y * Img.X_SIZE * 2 + A.x * 2 + 1] = ty;
		}
		else {
			P2 = Point2i(A.x, A.y);
			C = A + Sm.nn(P2, A.z);
			if (non != PATCHSIZEint3 && nowOCC.Image[C.z].data[C.y * Img.X_SIZE + C.x] == 0) {
				color = Img.Color.Image[C.z].at<Vec3b>(C.y, C.x);	// ピクセル値（カラー）を取得
				Img.Color.Image[A.z].at<Vec3b>(A.y, A.x) = color;
				tx = (float)Img.Texture.Image[C.z].data[C.y * Img.X_SIZE * 2 + C.x * 2 + 0];	// テクスチャ特徴を取得
				ty = (float)Img.Texture.Image[C.z].data[C.y * Img.X_SIZE * 2 + C.x * 2 + 1];
				Img.Texture.Image[A.z].data[A.y * Img.X_SIZE * 2 + A.x * 2 + 0] = tx;
				Img.Texture.Image[A.z].data[A.y * Img.X_SIZE * 2 + A.x * 2 + 1] = ty;

			}
			//else{ cout << "WARRNING: Spq is out size!" << endl; }
		}

		Cost.clear();
		sqrtCost.clear();
		colorR.clear();
		colorG.clear();
		colorB.clear();
		TX.clear();
		TY.clear();
	}
}
// Reconstruction of U&T
void Reconstruction(VideoSet& Img, ShiftMap& Sm, vector<Point3i>& inpaintP) {
	Patch PATCH;
	double shigma, Spq;
	vector<double> Cost, sqrtCost;
	Point3i A, B, C;
	Point2i P2;
	double costSUM, costSUMsqrt;
	vector<double> colorR, colorG, colorB;
	double avgColorR, avgColorG, avgColorB;
	Vec3b color;
	uchar r, g, b;
	vector<float> TX, TY;
	float tx, ty;
	double avgX, avgY;
	int non, P_SIZEint;
	for (int i = 0; i < inpaintP.size(); i++) {
		A = inpaintP[i];
		non = 0;
		for (int z = A.z + PATCHstart; z < A.z + PATCHend; z++) {
			for (int y = A.y + PATCHstart; y < A.y + PATCHend; y++) {
				for (int x = A.x + PATCHstart; x < A.x + PATCHend; x++) {
					if (x >= 0 && x < Img.X_SIZE && y >= 0 && y < Img.Y_SIZE && z >= 0 && z < Img.Z_SIZE) {
						if (Img.Occlusion.Image[z].data[y * Img.X_SIZE + x] == 0) {
							C = Point3i(x, y, z);
							P2 = Point2i(x, y);
							B = C + Sm.nn(P2, C.z);
							PATCH = Patch(C, B, Img.Color, Img.Texture, Img.Occlusion, Sm);
							C = A + Sm.nn(P2, C.z);
							if (C.x >= 0 && C.x < Img.X_SIZE && C.y >= 0 && C.y < Img.Y_SIZE && C.z >= 0 && C.z < Img.Z_SIZE) {
								Cost.push_back(PATCH.costfunction(3));
								sqrtCost.push_back(PATCH.costfunction(2));

								color = Img.Color.Image[C.z].at<Vec3b>(C.y, C.x);	// ピクセル値（カラー）を取得
								r = color[2];	// R,G,B値に分解
								g = color[1];
								b = color[0];
								colorR.push_back((double)r);
								colorG.push_back((double)g);
								colorB.push_back((double)b);
								tx = (float)Img.Texture.Image[C.z].data[C.y * Img.X_SIZE * 2 + C.x * 2 + 0];	// テクスチャ特徴を取得
								ty = (float)Img.Texture.Image[C.z].data[C.y * Img.X_SIZE * 2 + C.x * 2 + 1];
								TX.push_back(tx);
								TY.push_back(ty);
							}
							else { non++; }
							PATCH = Patch();
						}
						else { non++; }
					}
					else { non++; }
				}
			}
		}
		//cout << "  non = " << non << "  (/125)" << endl;	// 確認用

		// sort Cost and return 75 persentile
		shigma = SHIGMA(sqrtCost);
		Spq = 0.0;
		P_SIZEint = PATCHSIZEint * PATCHSIZEint * PATCHSIZEint - non;
		for (int ii = 0; ii < P_SIZEint; ii++) {
			Spq = Spq + (double)exp(-Cost[ii] / (2 * shigma * shigma));
		}
		//cout << "Spq： " << Spq << endl;	// 確認用

		if (Spq > 0 && Spq < 10000) {	// ※-nan(ind),1e+06に対する対応
			// reconstruct u(p)
			avgColorR = 0.0, avgColorG = 0.0, avgColorB = 0.0;
			for (int ii = 0; ii < P_SIZEint; ii++) {
				avgColorR = avgColorR + Spq * colorR[ii];
				avgColorG = avgColorG + Spq * colorG[ii];
				avgColorB = avgColorB + Spq * colorB[ii];
			}
			avgColorR = avgColorR / P_SIZEint;
			avgColorG = avgColorG / P_SIZEint;
			avgColorB = avgColorB / P_SIZEint;
			r = (uchar)(avgColorR / Spq);	// R,G,B値を処理
			g = (uchar)(avgColorG / Spq);
			b = (uchar)(avgColorB / Spq);
			//color = Img.Color.Image[A.z].at<Vec3b>(A.y, A.x);	// 確認用
			//cout << color << " -> ";
			color = Vec3b(b, g, r);				// R,G,B値からピクセル値（カラー）を生成
			Img.Color.Image[A.z].at<Vec3b>(A.y, A.x) = color;	// ピクセル値（カラー）を設定
			//cout << color << endl;	// 確認用

			// reconstruct t(p)
			avgX = 0.0, avgY = 0.0;
			for (int ii = 0; ii < P_SIZEint; ii++) {
				avgX = avgX + Spq * TX[ii];
				avgY = avgY + Spq * TY[ii];
			}
			avgX = avgX / P_SIZEint;
			avgY = avgY / P_SIZEint;
			tx = (float)(avgX / Spq);
			ty = (float)(avgY / Spq);
			Img.Texture.Image[A.z].data[A.y * Img.X_SIZE * 2 + A.x * 2 + 0] = tx;
			Img.Texture.Image[A.z].data[A.y * Img.X_SIZE * 2 + A.x * 2 + 1] = ty;
		}
		else {
			P2 = Point2i(A.x, A.y);
			C = A + Sm.nn(P2, A.z);
			if (non != PATCHSIZEint3 && Img.Occlusion.Image[C.z].data[C.y * Img.X_SIZE + C.x] == 0) {
				color = Img.Color.Image[C.z].at<Vec3b>(C.y, C.x);	// ピクセル値（カラー）を取得
				Img.Color.Image[A.z].at<Vec3b>(A.y, A.x) = color;
				tx = (float)Img.Texture.Image[C.z].data[C.y * Img.X_SIZE * 2 + C.x * 2 + 0];	// テクスチャ特徴を取得
				ty = (float)Img.Texture.Image[C.z].data[C.y * Img.X_SIZE * 2 + C.x * 2 + 1];
				Img.Texture.Image[A.z].data[A.y * Img.X_SIZE * 2 + A.x * 2 + 0] = tx;
				Img.Texture.Image[A.z].data[A.y * Img.X_SIZE * 2 + A.x * 2 + 1] = ty;

			}
			else {
				P2 = Point2i(A.x, A.y);
				C = A + Sm.nn(P2, A.z);
				if (C.x < 0 && C.x >= Img.X_SIZE && C.y < 0 && C.y >= Img.Y_SIZE && C.z < 0 && C.z >= frame) {
					//cout << "WARRNING: ShiftMap is out size!" << endl;
				}
				else if (non < PATCHSIZEint3 && Img.Occlusion.Image[C.z].data[C.y * Img.X_SIZE + C.x] == 0) {
					color = Img.Color.Image[C.z].at<Vec3b>(C.y, C.x);	// ピクセル値（カラー）を取得
					Img.Color.Image[A.z].at<Vec3b>(A.y, A.x) = color;
					tx = (float)Img.Texture.Image[C.z].data[C.y * Img.X_SIZE * 2 + C.x * 2 + 0];	// テクスチャ特徴を取得
					ty = (float)Img.Texture.Image[C.z].data[C.y * Img.X_SIZE * 2 + C.x * 2 + 1];
					Img.Texture.Image[A.z].data[A.y * Img.X_SIZE * 2 + A.x * 2 + 0] = tx;
					Img.Texture.Image[A.z].data[A.y * Img.X_SIZE * 2 + A.x * 2 + 1] = ty;

				}
				//else { cout << "WARRNING: Spq is out size! : Spq = " << Spq << endl; }
			}
		}

		Cost.clear();
		sqrtCost.clear();
		colorR.clear();
		colorG.clear();
		colorB.clear();
		TX.clear();
		TY.clear();
	}
}

// if the patch included occluded pixels, return the number of it
//int unoccluded_checker(Point3i center, Video& occluded) {
//	int number_occ = 0;
//	for (int z_p = center.z + PATCHstart; z_p < center.z + PATCHend; z_p++) {
//		for (int y_p = center.y + PATCHstart; y_p < center.y + PATCHend; y_p++) {
//			for (int x_p = center.x + PATCHstart; x_p < center.x + PATCHend; x_p++) {
//				if (x_p >= 0 && x_p < occluded.XSIZE && y_p >= 0 && y_p < occluded.YSIZE && z_p >= 0 && z_p < occluded.ZSIZE) {
//					if (occluded.Image[z_p].data[y_p * occluded.XSIZE + x_p] != 0) {
//						number_occ++;
//					}
//				}
//			}
//		}
//	}
//	return number_occ;
//}

double SHIGMA(vector<double>& array) {
	// sort Cost
	int c_num = 0;
	//vector<double> CostUP(array.size());	// 0含む
	//copy(array.begin(), array.end(), CostUP.begin());
	//sort(CostUP.begin(), CostUP.end());
	//c_num = array.size();
	vector<double> CostUP;	// 0含まない
	for (int array_index = 0; array_index < array.size(); array_index++) {
		if (array[array_index] != 0) {
			CostUP.push_back(array[array_index]);
			c_num++;
		}
	}
	/*if (c_num == array.size()) {
		cout << " ERROR: SHIGMA no fitting array!" << endl;
	}*/

	double Q3 = 100000.0;
	int Quartile;
	for (int i = 0; i < c_num; i++) {
		if (c_num % 2 == 0) {
			int half = c_num / 2;
			if (half % 2 == 0) {
				Quartile = (half - 1) + half / 2;
				Q3 = (double)(CostUP[Quartile] + CostUP[Quartile + 1]) / 2.0;
			}
			else {
				Quartile = (half - 1) + (half + 1) / 2;
				Q3 = (double)CostUP[Quartile];
			}
		}
		else {
			int half = (c_num - 1) / 2;
			if (half % 2 == 0) {
				Quartile = (half - 1) + half / 2;
				Q3 = (double)(CostUP[Quartile] + CostUP[Quartile + 1]) / 2.0;
			}
			else {
				Quartile = (half - 1) + (half + 1) / 2;
				Q3 = (double)CostUP[Quartile];
			}
		}
		//// Costとsort後の値確認
		//cout << "Cost" << endl;
		//for (int i = 0; i < c_num; i++) {
		//	cout << array[i] << endl;
		//}
		//cout << "Sort" << endl;
		//for (int i = 0; i < c_num; i++) {
		//	cout << CostUP[i] << endl;
		//}
		//// shigmaの値確認
		//cout << "shigma： " << Q3 << endl;
	}
	/*Q3 = cubeRoot(Q3);*/
	while (Q3 == 0) { Quartile++; Q3 = (double)CostUP[Quartile]; }
	/*if (Q3 == 0) { Q3 = 1; }*/
	CostUP.clear();
	return Q3;
}

// アニーリング（焼きなまし）
//void annealingReconstruction(LClass& Img, ShiftMap& Sm) {
//	int c_num = 0;
//	vector<int> occNUMx;
//	vector<int> occNUMy;
//	for (int y = 0; y < Img.imgU.rows; y++) {
//		for (int x = 0; x < Img.imgU.cols; x++) {
//			if (Img.imgO.data[y * Img.XSIZE + x] != 0) {
//				c_num++;
//				occNUMx.push_back(x);
//				occNUMy.push_back(y);
//			}
//		}
//	}
//
//	double COST, COST2, minCOST;
//	double Cost, Cost2, minCost;
//	Point2i A, B, C, Pix;
//	Vec3b color;
//	int TX, TY;
//	for (int i = 0; i < c_num; i++) {
//		A.x = occNUMx[i];
//		A.y = occNUMy[i];
//		minCOST = DBL_MAX; // 最大値を代入
//		for (int y = A.y + PATCHstart; y < A.y + PATCHend; y++) {
//			for (int x = A.x + PATCHstart; x < A.x + PATCHend; x++) {
//				if (x >= 0 && x < Img.XSIZE && y >= 0 && y < Img.YSIZE) {
//					C = Point2i(x, y);
//					B.x = C.x + Sm.nnX(C);
//					B.y = C.y + Sm.nnY(C);
//					COST = CostFunctionF(C, B, Img.imgU, Img.imgT, Img.imgO);
//					COST2 = CostFunctionFsqrt(C, B, Img.imgU, Img.imgT, Img.imgO);
//					//cout << "     COST: " << COST << " COST2: " << COST2 << endl;
//					if (minCOST > COST) {
//						minCOST = COST;
//						Pix = Point2i(A.x + Sm.nnX(C), A.y + Sm.nnY(C));
//						color = Img.imgU.at<Vec3b>(Pix.y, Pix.x);	// ピクセル値（カラー）を取得
//						TX = Img.imgT.data[Pix.y * Img.XSIZE * 2 + Pix.x * 2];	// テクスチャ特徴を取得
//						TY = Img.imgT.data[Pix.y * Img.XSIZE * 2 + Pix.x * 2 + 1];
//						//cout << "      min cost : " << minCOST << Pix << Point << endl;
//					}
//				}
//				else { /*cout << "B does not include " << Point2i(xp, yp) << endl;*/ }
//			}
//		}
//		///* 確認用 */
//		//cout << "Final : " << Point << " <- " << Pix << "  : Cost " << minCOST << endl;
//		//cout << " colorR: " << (int)color[2] << "  colorG: " << (int)color[1] << "  colorB: " << (int)color[0] << endl;
//		Img.imgU.at<Vec3b>(A.y, A.x) = color;	// ピクセル値（カラー）を設定
//		Img.imgT.data[A.y * Img.XSIZE * 2 + A.x * 2] = TX;	// テクスチャを設定
//		Img.imgT.data[A.y * Img.XSIZE * 2 + A.x * 2 + 1] = TY;
//	}
//	occNUMx.clear();
//	occNUMy.clear();
//}

// 部分的ノイズ除去 withピラミッド情報(img_dst2に出力)
void Gamma_OCC_MRF_GaussSeidel_Color(VideoPyramid& videoPyr) {
	// パラメータ設定
	vector<double> GAMMA;
	GAMMA.push_back(0.3);	//Level 0
	GAMMA.push_back(0.3);	//Level 1
	GAMMA.push_back(0.3);	//Level 2

	double errorConvergence;
	double number[3];
	double denom, ave[3];
	double M_number[3];	// オクルージョン境界部
	double D_number[3];	// オクルージョン内部
	double Yi[3];

	Vec3b color;
	uchar r, g, b;
	vector<Mat> RandomMap_R, RandomMap_G, RandomMap_B;
	vector<Mat> RandomMap_temp;
	Mat Map_R, Map_G, Map_B;
	for (int T = 0; T < frame; T++) {
		for (int i_pyr = 0; i_pyr <= L; i_pyr++) {
			Map_R = Mat(videoPyr.nLEVEL_video[i_pyr].Color.Image[T].rows, videoPyr.nLEVEL_video[i_pyr].Color.Image[T].cols, CV_64FC3, Scalar::all(0.5));
			Map_G = Mat(videoPyr.nLEVEL_video[i_pyr].Color.Image[T].rows, videoPyr.nLEVEL_video[i_pyr].Color.Image[T].cols, CV_64FC3, Scalar::all(0.5));
			Map_B = Mat(videoPyr.nLEVEL_video[i_pyr].Color.Image[T].rows, videoPyr.nLEVEL_video[i_pyr].Color.Image[T].cols, CV_64FC3, Scalar::all(0.5));
			RandomMap_R.push_back(Map_B);
			RandomMap_G.push_back(Map_G);
			RandomMap_B.push_back(Map_R);
		}
		Mat Image_dst;
		videoPyr.nLEVEL_video[0].Color.Image[T].copyTo(Image_dst);
		Image_dst.copyTo(img_dst2[T]);

		// オクルージョンピクセルの抽出
		vector<int> occ_num;
		int occ_number;
		vector<Point2i> Occ_Pixel, storeOcc_Pixel;
		vector<int> up;
		vector<int> down;
		vector<int> left;
		vector<int> right;
		vector<int> pyr_up;
		vector<int> pyr_down;
		int index;
		int h_x, h_y, h_index, h_Level;
		for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
			occ_number = 0;
			for (int Y_index = 0; Y_index < videoPyr.nLEVEL_Y_SIZE[int_pyr]; Y_index++) {
				for (int X_index = 0; X_index < videoPyr.nLEVEL_X_SIZE[int_pyr]; X_index++) {
					index = Y_index * videoPyr.nLEVEL_X_SIZE[int_pyr] + X_index;
					if (videoPyr.nLEVEL_video[int_pyr].Occlusion.Image[T].data[index] != 0) {
						Occ_Pixel.push_back(Point2i(X_index, Y_index));
						if(int_pyr == 0){ storeOcc_Pixel.push_back(Point2i(X_index, Y_index)); }
						occ_number++;

						// 隣接データは「データ外 =0」か「境界部 M=1」か「内部 D=2」に含まれる
						if (X_index - 1 < 0) { left.push_back(0); }				// left
						else if (videoPyr.nLEVEL_video[int_pyr].Occlusion.Image[T].data[index - 1] == 0) { left.push_back(1); }
						else { left.push_back(2); }
						if (Y_index - 1 < 0) { up.push_back(0); }					// up
						else if (videoPyr.nLEVEL_video[int_pyr].Occlusion.Image[T].data[index - videoPyr.nLEVEL_video[int_pyr].X_SIZE] == 0) { up.push_back(1); }
						else { up.push_back(2); }
						if (X_index + 1 >= videoPyr.nLEVEL_video[int_pyr].X_SIZE) { right.push_back(0); }	// right
						else if (videoPyr.nLEVEL_video[int_pyr].Occlusion.Image[T].data[index + 1] == 0) { right.push_back(1); }
						else { right.push_back(2); }
						if (Y_index + 1 >= videoPyr.nLEVEL_video[int_pyr].Y_SIZE) { down.push_back(0); }	//down
						else if (videoPyr.nLEVEL_video[int_pyr].Occlusion.Image[T].data[index + videoPyr.nLEVEL_video[int_pyr].X_SIZE] == 0) { down.push_back(1); }
						else { down.push_back(2); }

						h_Level = int_pyr + 1;			//pyramid up
						if (h_Level > L) { pyr_up.push_back(0); }
						else {
							if (X_index % 2 == 0) { h_x = X_index / 2; }
							else { h_x = (X_index - 1) / 2; }
							if (Y_index % 2 == 0) { h_y = Y_index / 2; }
							else { h_y = (Y_index - 1) / 2; }
							h_index = h_y * videoPyr.nLEVEL_video[h_Level].Occlusion.Image[T].cols + h_x;
							if (videoPyr.nLEVEL_video[h_Level].Occlusion.Image[T].data[h_index] != 0) { pyr_up.push_back(2); }
							else { pyr_up.push_back(1); }
						}
						h_Level = int_pyr - 1;			//pyramid down
						if (h_Level < 0) { pyr_down.push_back(0); }
						else {
							h_x = X_index * 2;	//対応する４マスの内単純に２倍のものに対応付け
							h_y = Y_index * 2;
							h_index = h_y * videoPyr.nLEVEL_X_SIZE[int_pyr] + h_x;
							if (videoPyr.nLEVEL_video[h_Level].Occlusion.Image[T].data[h_index] != 0) { pyr_down.push_back(2); }
							else { pyr_down.push_back(1); }
						}
					}
				}
			}
			occ_num.push_back(occ_number);
		}

		// RGB値からxを決める
		int pix_X, pix_Y;
		for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
			for (int Y_index = 0; Y_index < videoPyr.nLEVEL_Y_SIZE[int_pyr]; Y_index++) {
				for (int X_index = 0; X_index < videoPyr.nLEVEL_X_SIZE[int_pyr]; X_index++) {
					pix_X = X_index;
					pix_Y = Y_index;

					color = videoPyr.nLEVEL_video[int_pyr].Color.Image[T].at<Vec3b>(pix_Y, pix_X);	// ピクセル値（カラー）を取得
					r = color[2];	// R,G,B値に分解
					g = color[1];
					b = color[0];

					Yi[2] = (double)r / (double)MAX_INTENSE;
					Yi[1] = (double)g / (double)MAX_INTENSE;
					Yi[0] = (double)b / (double)MAX_INTENSE;
					RandomMap_R[int_pyr].at<double>(pix_Y, pix_X) = (double)Yi[2];
					RandomMap_G[int_pyr].at<double>(pix_Y, pix_X) = (double)Yi[1];
					RandomMap_B[int_pyr].at<double>(pix_Y, pix_X) = (double)Yi[0];
				}
			}
		}

		// ノイズ除去
		double Gamma_index;
		int occ_index;
		for (int index_R = 0; index_R < Repeat; index_R++) {
			errorConvergence = 0;
			occ_index = 0;
			occ_number = 0;
			for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
				Gamma_index = GAMMA[int_pyr];
				//cout << (int)int_pyr << " -> Gamma:" << Gamma[int_pyr] << endl;

				RandomMap_temp.push_back(RandomMap_B[int_pyr]);
				RandomMap_temp.push_back(RandomMap_G[int_pyr]);
				RandomMap_temp.push_back(RandomMap_R[int_pyr]);
				occ_number += occ_num[int_pyr];

				for (int Y_index = 0; Y_index < videoPyr.nLEVEL_Y_SIZE[int_pyr]; Y_index++) {
					for (int X_index = 0; X_index < videoPyr.nLEVEL_X_SIZE[int_pyr]; X_index++) {
						pix_X = X_index;
						pix_Y = Y_index;

						color = videoPyr.nLEVEL_video[int_pyr].Color.Image[T].at<Vec3b>(pix_Y, pix_X);	// ピクセル値（カラー）を取得
						r = color[2];	// R,G,B値に分解
						g = color[1];
						b = color[0];
						Yi[2] = (double)r / (double)MAX_INTENSE;
						Yi[1] = (double)g / (double)MAX_INTENSE;
						Yi[0] = (double)b / (double)MAX_INTENSE;

						index = pix_Y * videoPyr.nLEVEL_X_SIZE[int_pyr] + pix_X;

						for (int color_index = 0; color_index < 3; color_index++) {
							number[color_index] = (double)Yi[color_index];
						}
						denom = Ramda + 1;

						if (videoPyr.nLEVEL_video[int_pyr].Occlusion.Image[T].data[index] != 0) {
							for (int color_index = 0; color_index < 3; color_index++) {
								M_number[color_index] = 0;
								D_number[color_index] = 0;
							}
							if (left[occ_index] != 0) {		// left
								if (left[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X - 1);
									M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X - 1);
									M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X - 1);
								}
								else {
									D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X - 1);
									D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X - 1);
									D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X - 1);
								}
								denom += Gamma_index;
							}
							if (right[occ_index] != 0) {		// right
								if (right[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X + 1);
									M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X + 1);
									M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X + 1);
								}
								else {
									D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X + 1);
									D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X + 1);
									D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X + 1);
								}
								denom += Gamma_index;
							}
							if (up[occ_index] != 0) {		// up
								if (up[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y - 1, pix_X);
									M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y - 1, pix_X);
									M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y - 1, pix_X);
								}
								else {
									D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y - 1, pix_X);
									D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y - 1, pix_X);
									D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y - 1, pix_X);
								}
								denom += Gamma_index;
							}
							if (down[occ_index] != 0) {		// down
								if (down[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y + 1, pix_X);
									M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y + 1, pix_X);
									M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y + 1, pix_X);
								}
								else {
									D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y + 1, pix_X);
									D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y + 1, pix_X);
									D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y + 1, pix_X);
								}
								denom += Gamma_index;
							}

							if (pyr_up[occ_index] != 0) {		//pyramid up
								h_Level = int_pyr + 1;
								if (pix_X % 2 == 0) { h_x = pix_X / 2; }
								else { h_x = (pix_X - 1) / 2; }
								if (pix_Y % 2 == 0) { h_y = pix_Y / 2; }
								else { h_y = (pix_Y - 1) / 2; }
								/* 確認 */
								/*cout << int_pyr << "   " << pix_X << ":" << pix_Y << " -> " << h_x << ":" << h_y << endl;
								cout << pyr_up[occ_index] << endl;*/

								if (pyr_up[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
									M_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
									M_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								}
								else {
									D_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
									D_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
									D_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								}
								denom += Gamma_index;
							}
							if (pyr_down[occ_index] != 0) {		//pyramid down
								h_Level = int_pyr - 1;
								h_x = pix_X * 2;
								h_y = pix_Y * 2;

								if (pyr_down[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
									M_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
									M_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								}
								else {
									D_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
									D_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
									D_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								}
								denom += Gamma_index;
							}

							for (int color_index = 0; color_index < 3; color_index++) {
								number[color_index] += Gamma_index * (double)(M_number[color_index] + D_number[color_index]);
							}

							occ_index++;
							if (occ_index > occ_number) { cout << "WARNING! : occ_index > occ_number in " << int_pyr << endl; }
						}

						for (int color_index = 0; color_index < 3; color_index++) {
							ave[color_index] = (double)number[color_index] / (double)denom;
							if (color_index == 0) {
								errorConvergence += fabs(RandomMap_B[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
							}
							else if (color_index == 1) {
								errorConvergence += fabs(RandomMap_G[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
							}
							else if (color_index == 2) {
								errorConvergence += fabs(RandomMap_R[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
							}

							int temp_index = int_pyr * 3 + color_index;
							RandomMap_temp[temp_index].at<double>(pix_Y, pix_X) = (double)ave[color_index];
						}
						///* 確認 */
						//cout << errorConvergence << endl;
					}
				}
				errorConvergence = (double)(errorConvergence / 3);
			}

			int temp_index = 0;
			for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
				RandomMap_temp[temp_index].copyTo(RandomMap_B[int_pyr]);
				RandomMap_temp[temp_index + 1].copyTo(RandomMap_G[int_pyr]);
				RandomMap_temp[temp_index + 2].copyTo(RandomMap_R[int_pyr]);
				temp_index = temp_index + 3;
			}
			RandomMap_temp.clear();

			/*errorConvergence = (double)(errorConvergence / L);*/
			if (errorConvergence / MAX_DATA < Converge) {
				cout << "収束成功: errorConvergence = " << errorConvergence << " , Iteration " << index_R + 1 << endl;
				break;
			}
			/*else {
				cout << "収束失敗!: errorConvergence = " << errorConvergence << " , Iteration " << index_R << endl;
			}*/
		}

		// 画像補修
		int occ_X, occ_Y;
		occ_number = storeOcc_Pixel.size();
		cout << "MRF(" << T << "): OCC size = " << occ_number << endl;	// 確認用
		for (int occ_index = 0; occ_index < occ_number; occ_index++) {
			//cout << "MRF: OCC pix = " << storeOcc_Pixel[occ_index] << endl;	// 確認用
			occ_X = storeOcc_Pixel[occ_index].x;
			occ_Y = storeOcc_Pixel[occ_index].y;
		/*for (int occ_index = 0; occ_index < nonVSp.size(); occ_index++) {
			occ_X = nonVSp[occ_index].x;
			occ_Y = nonVSp[occ_index].y;*/

			ave[0] = (int)((double)RandomMap_B[0].at<double>(occ_Y, occ_X) * (double)MAX_INTENSE);
			ave[1] = (int)((double)RandomMap_G[0].at<double>(occ_Y, occ_X) * (double)MAX_INTENSE);
			ave[2] = (int)((double)RandomMap_R[0].at<double>(occ_Y, occ_X) * (double)MAX_INTENSE);
			for (int color_index = 0; color_index < 3; color_index++) {
				if (ave[color_index] < 0) {
					ave[color_index] = 0;
					cout << "WARNING! : under0" << Point2i(occ_X, occ_Y) << endl;
				}
				else if (ave[color_index] > 255) {
					ave[color_index] = 255;
					cout << "WARNING! : over255" << Point2i(occ_X, occ_Y) << endl;
				}
			}

			color[2] = (uchar)ave[2];	// R,G,B値に分解
			color[1] = (uchar)ave[1];
			color[0] = (uchar)ave[0];
			Image_dst.at<Vec3b>(occ_Y, occ_X) = color;	// ピクセル値（カラー）
		}
		Image_dst.copyTo(img_dst2[T]);

		RandomMap_R.clear();
		RandomMap_G.clear();
		RandomMap_B.clear();
		Occ_Pixel.clear();
		storeOcc_Pixel.clear();
		up.clear();
		down.clear();
		left.clear();
		right.clear();
		pyr_up.clear();
		pyr_down.clear();
	}
	GAMMA.clear();
}

void Gamma_OCC_MRF_GaussSeidel_Color(VideoPyramid& videoPyr, vector<Video>& nowOcclusion) {
	// パラメータ設定
	vector<double> GAMMA;
	GAMMA.push_back(0.3);	//Level 0
	GAMMA.push_back(0.3);	//Level 1
	GAMMA.push_back(0.3);	//Level 2

	double errorConvergence;
	double number[3];
	double denom, ave[3];
	double M_number[3];	// オクルージョン境界部
	double D_number[3];	// オクルージョン内部
	double Yi[3];

	Vec3b color;
	uchar r, g, b;
	vector<Mat> RandomMap_R, RandomMap_G, RandomMap_B;
	vector<Mat> RandomMap_temp;
	Mat Map_R, Map_G, Map_B;
	for (int T = 0; T < frame; T++) {
		for (int i_pyr = 0; i_pyr <= L; i_pyr++) {
			Map_R = Mat(videoPyr.nLEVEL_video[i_pyr].Color.Image[T].rows, videoPyr.nLEVEL_video[i_pyr].Color.Image[T].cols, CV_64FC3, Scalar::all(0.5));
			Map_G = Mat(videoPyr.nLEVEL_video[i_pyr].Color.Image[T].rows, videoPyr.nLEVEL_video[i_pyr].Color.Image[T].cols, CV_64FC3, Scalar::all(0.5));
			Map_B = Mat(videoPyr.nLEVEL_video[i_pyr].Color.Image[T].rows, videoPyr.nLEVEL_video[i_pyr].Color.Image[T].cols, CV_64FC3, Scalar::all(0.5));
			RandomMap_R.push_back(Map_B);
			RandomMap_G.push_back(Map_G);
			RandomMap_B.push_back(Map_R);
		}
		Mat Image_dst;
		videoPyr.nLEVEL_video[0].Color.Image[T].copyTo(Image_dst);
		Image_dst.copyTo(img_dst2[T]);

		// オクルージョンピクセルの抽出
		vector<int> occ_num;
		int occ_number;
		vector<Point2i> Occ_Pixel, storeOcc_Pixel;
		vector<int> up;
		vector<int> down;
		vector<int> left;
		vector<int> right;
		vector<int> pyr_up;
		vector<int> pyr_down;
		int index;
		int h_x, h_y, h_index, h_Level;
		for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
			occ_number = 0;
			for (int Y_index = 0; Y_index < videoPyr.nLEVEL_Y_SIZE[int_pyr]; Y_index++) {
				for (int X_index = 0; X_index < videoPyr.nLEVEL_X_SIZE[int_pyr]; X_index++) {
					index = Y_index * videoPyr.nLEVEL_X_SIZE[int_pyr] + X_index;
					if (nowOcclusion[int_pyr].Image[T].data[index] != 0) {
						Occ_Pixel.push_back(Point2i(X_index, Y_index));
						if (int_pyr == 0) { storeOcc_Pixel.push_back(Point2i(X_index, Y_index)); }
						occ_number++;

						// 隣接データは「データ外 =0」か「境界部 M=1」か「内部 D=2」に含まれる
						if (X_index - 1 < 0) { left.push_back(0); }				// left
						else if (nowOcclusion[int_pyr].Image[T].data[index - 1] == 0) { left.push_back(1); }
						else { left.push_back(2); }
						if (Y_index - 1 < 0) { up.push_back(0); }					// up
						else if (nowOcclusion[int_pyr].Image[T].data[index - videoPyr.nLEVEL_video[int_pyr].X_SIZE] == 0) { up.push_back(1); }
						else { up.push_back(2); }
						if (X_index + 1 >= videoPyr.nLEVEL_video[int_pyr].X_SIZE) { right.push_back(0); }	// right
						else if (nowOcclusion[int_pyr].Image[T].data[index + 1] == 0) { right.push_back(1); }
						else { right.push_back(2); }
						if (Y_index + 1 >= videoPyr.nLEVEL_video[int_pyr].Y_SIZE) { down.push_back(0); }	//down
						else if (nowOcclusion[int_pyr].Image[T].data[index + videoPyr.nLEVEL_video[int_pyr].X_SIZE] == 0) { down.push_back(1); }
						else { down.push_back(2); }

						h_Level = int_pyr + 1;			//pyramid up
						if (h_Level > L) { pyr_up.push_back(0); }
						else {
							if (X_index % 2 == 0) { h_x = X_index / 2; }
							else { h_x = (X_index - 1) / 2; }
							if (Y_index % 2 == 0) { h_y = Y_index / 2; }
							else { h_y = (Y_index - 1) / 2; }
							h_index = h_y * nowOcclusion[int_pyr].Image[T].cols + h_x;
							if (nowOcclusion[int_pyr].Image[T].data[h_index] != 0) { pyr_up.push_back(2); }
							else { pyr_up.push_back(1); }
						}
						h_Level = int_pyr - 1;			//pyramid down
						if (h_Level < 0) { pyr_down.push_back(0); }
						else {
							h_x = X_index * 2;	//対応する４マスの内単純に２倍のものに対応付け
							h_y = Y_index * 2;
							h_index = h_y * videoPyr.nLEVEL_X_SIZE[int_pyr] + h_x;
							if (nowOcclusion[int_pyr].Image[T].data[h_index] != 0) { pyr_down.push_back(2); }
							else { pyr_down.push_back(1); }
						}
					}
				}
			}
			occ_num.push_back(occ_number);
		}

		// RGB値からxを決める
		int pix_X, pix_Y;
		for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
			for (int Y_index = 0; Y_index < videoPyr.nLEVEL_Y_SIZE[int_pyr]; Y_index++) {
				for (int X_index = 0; X_index < videoPyr.nLEVEL_X_SIZE[int_pyr]; X_index++) {
					pix_X = X_index;
					pix_Y = Y_index;

					color = videoPyr.nLEVEL_video[int_pyr].Color.Image[T].at<Vec3b>(pix_Y, pix_X);	// ピクセル値（カラー）を取得
					r = color[2];	// R,G,B値に分解
					g = color[1];
					b = color[0];

					Yi[2] = (double)r / (double)MAX_INTENSE;
					Yi[1] = (double)g / (double)MAX_INTENSE;
					Yi[0] = (double)b / (double)MAX_INTENSE;
					RandomMap_R[int_pyr].at<double>(pix_Y, pix_X) = (double)Yi[2];
					RandomMap_G[int_pyr].at<double>(pix_Y, pix_X) = (double)Yi[1];
					RandomMap_B[int_pyr].at<double>(pix_Y, pix_X) = (double)Yi[0];
				}
			}
		}

		// ノイズ除去
		double Gamma_index;
		int occ_index;
		for (int index_R = 0; index_R < Repeat; index_R++) {
			errorConvergence = 0;
			occ_index = 0;
			occ_number = 0;
			for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
				Gamma_index = GAMMA[int_pyr];
				//cout << (int)int_pyr << " -> Gamma:" << Gamma[int_pyr] << endl;

				RandomMap_temp.push_back(RandomMap_B[int_pyr]);
				RandomMap_temp.push_back(RandomMap_G[int_pyr]);
				RandomMap_temp.push_back(RandomMap_R[int_pyr]);
				occ_number += occ_num[int_pyr];

				for (int Y_index = 0; Y_index < videoPyr.nLEVEL_Y_SIZE[int_pyr]; Y_index++) {
					for (int X_index = 0; X_index < videoPyr.nLEVEL_X_SIZE[int_pyr]; X_index++) {
						pix_X = X_index;
						pix_Y = Y_index;

						color = videoPyr.nLEVEL_video[int_pyr].Color.Image[T].at<Vec3b>(pix_Y, pix_X);	// ピクセル値（カラー）を取得
						r = color[2];	// R,G,B値に分解
						g = color[1];
						b = color[0];
						Yi[2] = (double)r / (double)MAX_INTENSE;
						Yi[1] = (double)g / (double)MAX_INTENSE;
						Yi[0] = (double)b / (double)MAX_INTENSE;

						index = pix_Y * videoPyr.nLEVEL_X_SIZE[int_pyr] + pix_X;

						for (int color_index = 0; color_index < 3; color_index++) {
							number[color_index] = (double)Yi[color_index];
						}
						denom = Ramda + 1;

						if (nowOcclusion[int_pyr].Image[T].data[index] != 0) {
							for (int color_index = 0; color_index < 3; color_index++) {
								M_number[color_index] = 0;
								D_number[color_index] = 0;
							}
							if (left[occ_index] != 0) {		// left
								if (left[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X - 1);
									M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X - 1);
									M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X - 1);
								}
								else {
									D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X - 1);
									D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X - 1);
									D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X - 1);
								}
								denom += Gamma_index;
							}
							if (right[occ_index] != 0) {		// right
								if (right[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X + 1);
									M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X + 1);
									M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X + 1);
								}
								else {
									D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y, pix_X + 1);
									D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y, pix_X + 1);
									D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y, pix_X + 1);
								}
								denom += Gamma_index;
							}
							if (up[occ_index] != 0) {		// up
								if (up[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y - 1, pix_X);
									M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y - 1, pix_X);
									M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y - 1, pix_X);
								}
								else {
									D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y - 1, pix_X);
									D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y - 1, pix_X);
									D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y - 1, pix_X);
								}
								denom += Gamma_index;
							}
							if (down[occ_index] != 0) {		// down
								if (down[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y + 1, pix_X);
									M_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y + 1, pix_X);
									M_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y + 1, pix_X);
								}
								else {
									D_number[2] += (double)RandomMap_R[int_pyr].at<double>(pix_Y + 1, pix_X);
									D_number[1] += (double)RandomMap_G[int_pyr].at<double>(pix_Y + 1, pix_X);
									D_number[0] += (double)RandomMap_B[int_pyr].at<double>(pix_Y + 1, pix_X);
								}
								denom += Gamma_index;
							}

							if (pyr_up[occ_index] != 0) {		//pyramid up
								h_Level = int_pyr + 1;
								if (pix_X % 2 == 0) { h_x = pix_X / 2; }
								else { h_x = (pix_X - 1) / 2; }
								if (pix_Y % 2 == 0) { h_y = pix_Y / 2; }
								else { h_y = (pix_Y - 1) / 2; }
								/* 確認 */
								/*cout << int_pyr << "   " << pix_X << ":" << pix_Y << " -> " << h_x << ":" << h_y << endl;
								cout << pyr_up[occ_index] << endl;*/

								if (pyr_up[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
									M_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
									M_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								}
								else {
									D_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
									D_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
									D_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								}
								denom += Gamma_index;
							}
							if (pyr_down[occ_index] != 0) {		//pyramid down
								h_Level = int_pyr - 1;
								h_x = pix_X * 2;
								h_y = pix_Y * 2;

								if (pyr_down[occ_index] == 1) {
									M_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
									M_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
									M_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								}
								else {
									D_number[2] += (double)RandomMap_R[h_Level].at<double>(h_y, h_x);
									D_number[1] += (double)RandomMap_G[h_Level].at<double>(h_y, h_x);
									D_number[0] += (double)RandomMap_B[h_Level].at<double>(h_y, h_x);
								}
								denom += Gamma_index;
							}

							for (int color_index = 0; color_index < 3; color_index++) {
								number[color_index] += Gamma_index * (double)(M_number[color_index] + D_number[color_index]);
							}

							occ_index++;
							if (occ_index > occ_number) { cout << "WARNING! : occ_index > occ_number in " << int_pyr << endl; }
						}

						for (int color_index = 0; color_index < 3; color_index++) {
							ave[color_index] = (double)number[color_index] / (double)denom;
							if (color_index == 0) {
								errorConvergence += fabs(RandomMap_B[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
							}
							else if (color_index == 1) {
								errorConvergence += fabs(RandomMap_G[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
							}
							else if (color_index == 2) {
								errorConvergence += fabs(RandomMap_R[int_pyr].at<double>(pix_Y, pix_X) - ave[color_index]);
							}

							int temp_index = int_pyr * 3 + color_index;
							RandomMap_temp[temp_index].at<double>(pix_Y, pix_X) = (double)ave[color_index];
						}
						///* 確認 */
						//cout << errorConvergence << endl;
					}
				}
				errorConvergence = (double)(errorConvergence / 3);
			}

			int temp_index = 0;
			for (int int_pyr = 0; int_pyr <= L; int_pyr++) {
				RandomMap_temp[temp_index].copyTo(RandomMap_B[int_pyr]);
				RandomMap_temp[temp_index + 1].copyTo(RandomMap_G[int_pyr]);
				RandomMap_temp[temp_index + 2].copyTo(RandomMap_R[int_pyr]);
				temp_index = temp_index + 3;
			}
			RandomMap_temp.clear();

			/*errorConvergence = (double)(errorConvergence / L);*/
			if (errorConvergence / MAX_DATA < Converge) {
				cout << "収束成功: errorConvergence = " << errorConvergence << " , Iteration " << index_R + 1 << endl;
				break;
			}
			/*else {
				cout << "収束失敗!: errorConvergence = " << errorConvergence << " , Iteration " << index_R << endl;
			}*/
		}

		// 画像補修
		int occ_X, occ_Y;
		occ_number = storeOcc_Pixel.size();
		cout << "MRF(" << T << "): OCC size = " << occ_number << endl;	// 確認用
		for (int occ_index = 0; occ_index < occ_number; occ_index++) {
			//cout << "MRF: OCC pix = " << storeOcc_Pixel[occ_index] << endl;	// 確認用
			occ_X = storeOcc_Pixel[occ_index].x;
			occ_Y = storeOcc_Pixel[occ_index].y;
			/*for (int occ_index = 0; occ_index < nonVSp.size(); occ_index++) {
				occ_X = nonVSp[occ_index].x;
				occ_Y = nonVSp[occ_index].y;*/

			ave[0] = (int)((double)RandomMap_B[0].at<double>(occ_Y, occ_X) * (double)MAX_INTENSE);
			ave[1] = (int)((double)RandomMap_G[0].at<double>(occ_Y, occ_X) * (double)MAX_INTENSE);
			ave[2] = (int)((double)RandomMap_R[0].at<double>(occ_Y, occ_X) * (double)MAX_INTENSE);
			for (int color_index = 0; color_index < 3; color_index++) {
				if (ave[color_index] < 0) {
					ave[color_index] = 0;
					cout << "WARNING! : under0" << Point2i(occ_X, occ_Y) << endl;
				}
				else if (ave[color_index] > 255) {
					ave[color_index] = 255;
					cout << "WARNING! : over255" << Point2i(occ_X, occ_Y) << endl;
				}
			}

			color[2] = (uchar)ave[2];	// R,G,B値に分解
			color[1] = (uchar)ave[1];
			color[0] = (uchar)ave[0];
			Image_dst.at<Vec3b>(occ_Y, occ_X) = color;	// ピクセル値（カラー）
		}
		Image_dst.copyTo(img_dst2[T]);

		RandomMap_R.clear();
		RandomMap_G.clear();
		RandomMap_B.clear();
		Occ_Pixel.clear();
		storeOcc_Pixel.clear();
		up.clear();
		down.clear();
		left.clear();
		right.clear();
		pyr_up.clear();
		pyr_down.clear();
	}
	GAMMA.clear();
}

/* MSE計算 */
void MSE(Video& Image_1, Video& Image_Occ) {
	double MSE1 = 0, MSE2 = 0;
	double tmp_1, tmp_2;
	int img_size, color_ind;
	int occCOUNT = 0;
	for (int t = 0; t < Image_1.ZSIZE; t++) {
		for (int i = 0; i < Image_1.YSIZE; i++) {
			for (int j = 0; j < Image_1.XSIZE; j++) {
				img_size = i * Image_1.YSIZE + j;
				if (Image_Occ.Image[t].data[img_size] != 0) {	// オクルージョンのみチェック
					tmp_1 = 0;
					tmp_2 = 0;
					color_ind = i * width * 3 + j * 3;
					for (int channel = 0; channel < 3; channel++) {
						tmp_1 = tmp_1 + pow((int)video[t].data[color_ind] - (int)orig_video[t].data[color_ind], 2.0);
						tmp_2 = tmp_2 + pow((int)Image_1.Image[t].data[color_ind] - (int)orig_video[t].data[color_ind], 2.0);
						//cout << " temp1 = " << (double)tmp_1 << " , temp2 = " << (double)tmp_2 << endl;	// 確認用
						color_ind++;
					}
					MSE1 = MSE1 + (double)tmp_1;
					MSE2 = MSE2 + (double)tmp_2;
					occCOUNT++;
				}
			}
		}
	}
	img_size = occCOUNT * 3;
	MSE1 = (double)MSE1 / (double)img_size;
	MSE2 = (double)MSE2 / (double)img_size;
	MSE1 = (double)sqrt(MSE1);
	MSE2 = (double)sqrt(MSE2);

	/* 計算結果表示 */
	cout << "--- MSE -------" << endl;
	cout << "　Image 1 = " << (double)MSE1 << endl;
	cout << "　Image 2 = " << (double)MSE2 << endl;
}