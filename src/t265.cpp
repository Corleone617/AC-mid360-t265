#include <nav_msgs/Odometry.h>
#include <iostream>
#include <serial/serial.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <iomanip>
#include <sstream>
#include "fcu_serial/cifar.h"
#include "fcu_serial/AC_staue.h"

void camera_data_callback(const fcu_serial::cifar::ConstPtr & camera_data,float* flag,float* x_err,float* y_err,float* LeftorRight)
{
	*flag = camera_data->flag;
	*x_err = camera_data->x_err;
	*y_err = camera_data->y_err;
	*LeftorRight = camera_data->leftORright;
}

void position_call_back(const nav_msgs::Odometry::ConstPtr& odom_msg, double* pos_x, double* pos_y,double* pos_z,double* ori_w,double* ori_x,double* ori_y,double* ori_z, bool* rc_flag)
{
	*pos_x = odom_msg->pose.pose.position.x*100;
	*pos_y = odom_msg->pose.pose.position.y*100;
	*pos_z = odom_msg->pose.pose.position.z*100;
	*ori_x = odom_msg->pose.pose.orientation.x;
	*ori_y = odom_msg->pose.pose.orientation.y;
    *ori_z = odom_msg->pose.pose.orientation.z;
	*ori_w = odom_msg->pose.pose.orientation.w;
	*rc_flag = true;
}

void receiveSerialData(serial::Serial& serial_port,fcu_serial::AC_staue& AC_data)
{
    // 数据长度为5
    const int DATA_LENGTH = 5;

    // 检查缓冲区中的数据是否足够
    while (serial_port.available() >= DATA_LENGTH)
    {
        // 预读取固定长度的数据
        std::string received_data = serial_port.read(DATA_LENGTH);

        // 检查数据是否符合预定格式
        if (received_data.length() == DATA_LENGTH && received_data[0] == 'A' 
                                                  && received_data[1] == 'C'
                                                  && received_data[4] == 'D')
        {
			AC_data.waypoint = received_data[2];
			AC_data.upORdown = received_data[3];

            // ROS_INFO("%c  %c  %d  %d   %c", received_data[0], received_data[1], 
            //         static_cast<unsigned char>(received_data[2]), 
            //         static_cast<unsigned char>(received_data[3]), 
            //         received_data[4]);
        }
        else
        {
            // 数据不符合预期，清空缓冲区，避免错误累积
            serial_port.flushInput();
            ROS_WARN("Invalid data received, buffer flushed.");
        }
    }
}

static uint16_t calcCheckSum(uint8_t* buff)
{
	uint8_t sumA = 0, sumB=0;
	uint16_t msg_length = buff[3];
	for(uint8_t i=2; i<4+msg_length; ++i)
	{
		sumA += buff[i];
		sumB += sumA;
	}
	buff[4+msg_length + 0] = sumA;
	buff[4+msg_length + 1] = sumB;
	return 4+msg_length + 2;
}

int main (int argc, char** argv)
{
	ros::init(argc, argv, "t265_node");
	ros::NodeHandle nh;
	ros::Rate loop_rate(30);
 
	serial::Serial fcu_serial;
	double pos_x,pos_y,pos_z;
	double ori_x,ori_y,ori_z,ori_w;
	bool rc_flag = false;

	float flag = 0;
	float x_err = 0;
	float y_err = 0;
	float LeftorRight = 0;
	
	uint8_t tick = 0;
	
    ros::Subscriber self_sub  = nh.subscribe<fcu_serial::cifar>("/camera_data", 100,boost::bind(&camera_data_callback,_1,&flag,&x_err,&y_err,&LeftorRight));
 	// ros::Subscriber odom_sub1 = nh.subscribe<nav_msgs::Odometry>("/camera/odom/sample",100,boost::bind(&position_call_back,_1,&pos_x,&pos_y,&pos_z,&ori_w,&ori_x,&ori_y,&ori_z,&rc_flag));
	ros::Subscriber odom_sub2 = nh.subscribe<nav_msgs::Odometry>("/Odometry",100,boost::bind(&position_call_back,_1,&pos_x,&pos_y,&pos_z,&ori_w,&ori_x,&ori_y,&ori_z,&rc_flag));

	ros::Publisher AC_pub = nh.advertise<fcu_serial::AC_staue>("AC_data",100);

	fcu_serial.setPort("/dev/ttyUSB0");
 	fcu_serial.setBaudrate(115200);
	serial::Timeout to = serial::Timeout::simpleTimeout(1000);
	fcu_serial.setTimeout(to);

	try
	{
		// sudo chmod 777 /dev/ttyUSB0
		fcu_serial.open();
	}
	catch(const serial::IOException& e)
	{
		ROS_INFO_STREAM("Failed to open serial");
		return -1;
	}

	if(fcu_serial.isOpen()) {
		ROS_INFO_STREAM("serial opened");
	}
	else {
		ROS_INFO_STREAM("Failed to open serial");
		return -1;
	}

	fcu_serial::AC_staue AC_data;

	while(ros::ok())
	{
		if(++tick == 20)
		{
			receiveSerialData(fcu_serial,AC_data);  // 调用函数以接收数据
			AC_pub.publish(AC_data);
			tick = 0;
		}
		
		if(rc_flag)
		{
			uint8_t buffer[100];
			uint16_t ind = 0;
			buffer[ind++] = 'A';
			buffer[ind++] = 'C';
			uint8_t i=0;

			buffer[ind++] = 2;
			buffer[ind++] = 44;

			*((float*)&(buffer[ind])) = pos_x;
			ind += 4;
			*((float*)&(buffer[ind])) = pos_y;
			ind += 4;
			*((float*)&(buffer[ind])) = pos_z;
			ind += 4;
			*((float*)&(buffer[ind])) = ori_w;
			ind += 4;
			*((float*)&(buffer[ind])) = ori_x;
			ind += 4;
			*((float*)&(buffer[ind])) = ori_y;
			ind += 4;
			*((float*)&(buffer[ind])) = ori_z;
			ind += 4;
			// opencv数据
			*((float*)&(buffer[ind])) = flag;
			ind += 4;
            *((float*)&(buffer[ind])) = x_err;
			ind += 4;
            *((float*)&(buffer[ind])) = y_err;
			ind += 4;
            *((float*)&(buffer[ind])) = LeftorRight;
			ind += 4;

			uint16_t len = calcCheckSum(buffer);
			// ROS_INFO("x:%f y:%f ",pos_x,pos_y);

			fcu_serial.write(buffer,len);
			rc_flag = false;
		}
		
		ros::spinOnce();
		loop_rate.sleep();
	}
}