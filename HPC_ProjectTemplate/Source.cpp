#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <msclr\marshal_cppstd.h>
#include <ctime>// include this header 
#pragma once

#using <mscorlib.dll>
#using <System.dll>
#using <System.Drawing.dll>
#using <System.Windows.Forms.dll>
using namespace std;
using namespace msclr::interop;

int* inputImage(int* w, int* h, System::String^ imagePath) //put the size of image in w & h
{
	int* input;


	int OriginalImageWidth, OriginalImageHeight;

	//*********************************************************Read Image and save it to local arrayss*************************	
	//Read Image and save it to local arrayss

	System::Drawing::Bitmap BM(imagePath);

	OriginalImageWidth = BM.Width;
	OriginalImageHeight = BM.Height;
	*w = BM.Width;
	*h = BM.Height;
	int* Red = new int[BM.Height * BM.Width];
	int* Green = new int[BM.Height * BM.Width];
	int* Blue = new int[BM.Height * BM.Width];
	input = new int[BM.Height * BM.Width];
	for (int i = 0; i < BM.Height; i++)
	{
		for (int j = 0; j < BM.Width; j++)
		{
			System::Drawing::Color c = BM.GetPixel(j, i);

			Red[i * BM.Width + j] = c.R;
			Blue[i * BM.Width + j] = c.B;
			Green[i * BM.Width + j] = c.G;

			input[i * BM.Width + j] = ((c.R + c.B + c.G) / 3); //gray scale value equals the average of RGB values

		}

	}
	return input;
}


void createImage(int* image, int width, int height, int index)
{
	System::Drawing::Bitmap MyNewImage(width, height);


	for (int i = 0; i < MyNewImage.Height; i++)
	{
		for (int j = 0; j < MyNewImage.Width; j++)
		{
			//i * OriginalImageWidth + j
			if (image[i * width + j] < 0)
			{
				image[i * width + j] = 0;
			}
			if (image[i * width + j] > 255)
			{
				image[i * width + j] = 255;
			}
			System::Drawing::Color c = System::Drawing::Color::FromArgb(image[i * MyNewImage.Width + j], image[i * MyNewImage.Width + j], image[i * MyNewImage.Width + j]);
			MyNewImage.SetPixel(j, i, c);
		}
	}
	MyNewImage.Save("..//Data//Output//outputRes" + index + ".png");
	cout << "result Image Saved " << index << endl;
}


int main()
{
	int ImageWidth = 4, ImageHeight = 4;

	int start_s, stop_s, TotalTime = 0;

	System::String^ imagePath;
	std::string img;
	img = "..//Data//Input//test.png";

	imagePath = marshal_as<System::String^>(img);
	int* imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);
	
	start_s = clock();

	MPI_Init(NULL, NULL);
	int rank, size;
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int NewSize = (ImageWidth * ImageHeight) / size;
	int smallSize = 256 / size;
	int* localsum = new int[256];
	int* localsumTmp = new int[256];
	int* localimg = new int[NewSize];
	int globalsum[256] = { 0 };

	for (int i = 0; i < 256; i++)
	{		
		localsum[i] = 0;
		localsumTmp[i] = 0;
		globalsum[i] = 0;
	}

	double globalsum2[256] = { 0 };
	int localTotalSum = 0;
	int globalTotalSum = 0;

	double* localsum2 = new double[256];
	double probability[256] = { 0 };

	double* temp = new double[smallSize];
	double NewglobalTotalSum = 0;

	double commulative[256] = { 0 };
	int intscalling[256] = { 0 };
	int* inttemp = new int[smallSize];

	// -- Calculate Recurrence Of Image Values -- //
	MPI_Scatter(imageData, NewSize, MPI_INT, localimg, NewSize, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < NewSize; i++)
		localsum[localimg[i]]++;

	MPI_Reduce(localsum, &globalsum, 256, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	


	// -- Calculate the probability of each pixel intensity in the image matrix -- //
	MPI_Scatter(&globalsum, smallSize, MPI_INT, localsumTmp, smallSize, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < smallSize; i++)
		temp[i] = (double)localsumTmp[i] / (double)(ImageWidth*ImageHeight);

	MPI_Gather(temp, smallSize, MPI_DOUBLE, &probability, smallSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	if (rank == 0)
	{
		// -- Calculate commulative probability -- //
		commulative[0] = probability[0];
		for (int i = 1; i < 256; i++)
			commulative[i] = probability[i] + commulative[i - 1];
	}


	// -- Calculate pixels value scalling -- //
	MPI_Scatter(&commulative, smallSize, MPI_DOUBLE, localsum2, smallSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	for (int i = 0; i < smallSize; i++)
	{
		temp[i] = localsum2[i] * 256;
		inttemp[i] = floor(temp[i]);
	}
	
	MPI_Gather(inttemp, smallSize, MPI_INT, intscalling, smallSize, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(intscalling, 256, MPI_INT, 0, MPI_COMM_WORLD);
	
	
	
	// -- Update Image -- //
	for (int i = 0; i < NewSize; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			if (localimg[i] == j)
			{
				localimg[i] = intscalling[j];
				break;
			}
		}
	}

	MPI_Gather(localimg, NewSize, MPI_INT, imageData, NewSize, MPI_INT, 0, MPI_COMM_WORLD);
	

    // -- Create Image -- //
	if (rank == 0)
		createImage(imageData, ImageWidth, ImageHeight, 22);


	MPI_Finalize();		
	stop_s = clock();
	TotalTime += (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
	cout << "time: " << TotalTime << endl;

	free(imageData);
	return 0;
}
