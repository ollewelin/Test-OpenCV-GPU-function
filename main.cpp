#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::cuda;


int main()
{
    printf("GPU ON = %d\n", cv::cuda::getCudaEnabledDeviceCount());
    printf("OpenCV version:\n");
    std::cout << CV_VERSION << std::endl;


    printf("Now use file:\n");
    printf("input.JPG\n");

    cv::Mat input_jpg_BGR;
    input_jpg_BGR = cv::imread("input.JPG", 1);

    cv::cuda::HostMem input_jpg_FC3(input_jpg_BGR.rows, input_jpg_BGR.cols, CV_32FC3, cv::cuda::HostMem::PAGE_LOCKED);///Allocate a CPU memory PAGE_LOCKED is 2x faster transfer to GPU and HostMem suitable for stream Asynchronous data transfer
    cv::cuda::HostMem part_of_inputJPG(8,8,CV_32FC3, cv::cuda::HostMem::PAGE_LOCKED);///Allocate a CPU memory PAGE_LOCKED is 2x faster transfer to GPU and HostMem suitable for stream Asynchronous data transfer

    input_jpg_BGR.convertTo(input_jpg_FC3, CV_32FC3, 1.0f/255.0f);///Ordinary CPU function

    cv::cuda::Stream stream_data_A;///Instantiate stream object used for asynchronous operation

    cv::cuda::GpuMat test_gpu_mat;///
    cv::Mat m_test;
    cv::Mat m_sum_test;
    m_sum_test.create(2,2,CV_32FC3);
    cv::cuda::GpuMat sum_test;
    sum_test.create(2,2,CV_32FC3);
    for(int i=0;i<2;i++)
    {
        for(int j=0;j<2*3;j++)
        {
            m_sum_test.at<float>(i,j) = 3.0f * ((j%3)+1);
        }
    }
    sum_test.upload(m_sum_test);
    cv::Scalar sum_of_gpu_mat;
    cv::cuda::GpuMat gpu_roi_part;
    cv::cuda::GpuMat gpu_roi_part_B;
    cv::cuda::GpuMat gpu_roi_part_C;
    cv::cuda::GpuMat gpu_std_test;
    m_test.create(8,8,CV_8UC1);
    gpu_std_test.create(8,8,CV_8UC1);
    gpu_std_test.upload(m_test);
    cv::Scalar mean_from_g_mat;
    cv::Scalar std_deviation;
    if(input_jpg_FC3.cols > 49 && input_jpg_FC3.rows > 49)///Check proper size of image
    {
        printf("Start a GPU test\n");


        test_gpu_mat.create(input_jpg_FC3.rows,input_jpg_FC3.cols,CV_32FC3);

        gpu_roi_part.create(8,8,CV_32FC3);
        gpu_roi_part_B.create(8,8,CV_32FC3);
        gpu_roi_part_C.create(8,8,CV_32FC3);


        part_of_inputJPG.create(8,8,CV_32FC3);
        ///src(Rect(left,top,width, height)).copyTo(dst);
        for(int sync_async_mul = 0; sync_async_mul < 2; sync_async_mul++)
        {
            if(sync_async_mul == 0)
            {
                printf("Now test Synchronous multiply will block CPU\n");
            }
            else
            {
                printf("Now test Operation multiply Asynchronous with CPU\n");
            }

            for(int i=0; i<10; i++)
            {

                ///Synchronous GPU process. This Block the CPU and will waiting until GPU task finish.
                ///     test_gpu_mat.upload(input_jpg_FC3, NULL);///Data to GPU "device"

                ///Asynchronous GPU process. This free the CPU when GPU working on a task
                test_gpu_mat.upload(input_jpg_FC3, stream_data_A);///Data to GPU "device" stream will not block CPU here until test_gpu_mat.upload() finish
                test_gpu_mat(Rect(5+i,10,8,8)).copyTo(gpu_roi_part);///Inside NVIDIA Rect() a part of image in GPU to another GpuMat
                test_gpu_mat(Rect(7+i,26+i,8,8)).copyTo(gpu_roi_part_B);///Inside NVIDIA Rect() a part of image in GPU to another GpuMat
                ///input.JPG test was 50x50
                test_gpu_mat(Rect(10+i,40,8,8)).copyTo(gpu_roi_part_C);///Inside NVIDIA Rect() a part of image in GPU to another GpuMat

                float scaler = (float) i/10;
                if(sync_async_mul == 0)
                {
                    cv::cuda::multiply(gpu_roi_part,gpu_roi_part_B,gpu_roi_part_C, scaler);///Synchronous multiply will block CPU
                }
                else
                {
                    cv::cuda::multiply(gpu_roi_part,gpu_roi_part_B,gpu_roi_part_C, scaler, -1, stream_data_A);///Operation Asynchronous with CPU
                }
///cv::cuda::meanStdDev(const GpuMat& mtx, Scalar& mean, Scalar& stddev)

///cv::cuda::meanStdDev(gpu_std_test, mean_from_g_mat, std_deviation);

sum_of_gpu_mat = cv::cuda::sum(sum_test);
                gpu_roi_part_C.download(part_of_inputJPG, stream_data_A);///Data back to CPU "host"
                test_gpu_mat.download(input_jpg_FC3, stream_data_A);///Data back to CPU "host"

                ///Here we could do CPU code while GPU process data
                ///bool 	queryIfComplete () const  	Returns true if the current stream queue is finished.

                int CPU_free_time_counter = 0;
                while(stream_data_A.queryIfComplete() == false)
                {
                    CPU_free_time_counter++;
                    if(CPU_free_time_counter < 0)
                    {
                        CPU_free_time_counter--;///Stop count up if wrap over to negative sign;
                    }
                }
                stream_data_A.waitForCompletion();///void waitForCompletion(); Blocks the current CPU thread until all operations in the stream are complete.
                ///Now GPU operation finish

                printf("CPU_free_time_counter = %d\n", CPU_free_time_counter);

                imshow("input_jpg_FC3", input_jpg_FC3);
                imshow("part_of_inputJPG", part_of_inputJPG);

                waitKey(100);
            }
        }
        std::cout << sum_of_gpu_mat << std::endl;
      //  printf("sum_of_gpu_mat = %f\n", sum_of_gpu_mat);
        printf("sum_of_gpu_mat.channels() = %d\n", sum_of_gpu_mat.channels);
        float ch0_a = sum_of_gpu_mat[0];
        float ch1_a = sum_of_gpu_mat[1];
        float ch2_a = sum_of_gpu_mat[2];
        printf("ch0_a = %f\n", ch0_a);
        printf("ch1_a = %f\n", ch1_a);
        printf("ch2_a = %f\n", ch2_a);

        ///End GPU test
    }

    cv::Mat m_input_jpg_FC3;///
    m_input_jpg_FC3 = input_jpg_FC3.createMatHeader();///Copy over the ordinary OpenCV Mat if you want to use that.
    ///Note: now after the .createMatHeader() function used the m_input_jpg_FC3 and input_jpg_FC3 are the same physical memory on the CPU host
    for(int i=0;i<(m_input_jpg_FC3.cols);i++)
    {
        m_input_jpg_FC3.at<float>(0,i*3+0) = 0.0;///Blue
        m_input_jpg_FC3.at<float>(0,i*3+1) = 0.0;///Green
        m_input_jpg_FC3.at<float>(0,i*3+2) = 1.0;///Red
        ///Note this will also affect the input_jpg_FC3 because the m_input_jpg_FC3 and input_jpg_FC3 are the same physical memory on the CPU host
        ///So when add some RED pixel on first row they also aper in the cv::cuda::HostMem input_jpg_FC3 as well because they are same physical memory on the CPU host
    }
    m_input_jpg_FC3.at<float>(0,0) = 1.0;
    imshow("m_input_jpg_FC3", m_input_jpg_FC3);
    imshow("input_jpg_FC3", input_jpg_FC3);

    printf("End GPU test\n");
    waitKey(15000);
    input_jpg_BGR.release();
    m_input_jpg_FC3.release();
    input_jpg_FC3.release();
    part_of_inputJPG.release();
    test_gpu_mat.release();
    gpu_roi_part.release();
    gpu_roi_part_B.release();
    gpu_roi_part_C.release();///
    stream_data_A.~Stream();
    printf("End Program");
}
