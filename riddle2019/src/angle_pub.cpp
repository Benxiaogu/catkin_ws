#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <iconv.h>

/* for ROS and C++*/
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int16.h>
#include <std_msgs/String.h> 
#include <serial/serial.h>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
using namespace std;



#define FRAME_LEN	640 
#define	BUFFER_SIZE	4096

/* microphone mode */
#define eMicPhone_Closed 				0
#define eMicPhone_Activate 				1
#define eMicPhone_Communicate 			2
#define eMicPhone_Communicate_Quit 	   -2


/* UART COMMAND */
#define VERSION 		"VER\n"
#define RESET			"RESET\n"
#define TALK  			"CALL_MODE 1\n"
#define LOCALIZATION 	"CALL_MODE 0\n"
#define WAKE_UP_YES  	"WAKEUP 1\n"
#define WAKE_UP_NO  	"WAKEUP 0\n"
#define BEAM_0  		"BEAM 0\n"
#define BEAM_1  		"BEAM 1\n"
#define BEAM_2  		"BEAM 2\n"
#define BEAM_3  		"BEAM 3\n"
#define BEAM_4  		"BEAM 4\n"
#define BEAM_5  		"BEAM 5\n"







/* demo recognize the audio from microphone */



/* the mode of the microphone */
volatile int microphone_mode;
volatile int microphone_mode_cmd;


/* main thread: start/stop record ; query the result of recgonization.
 * record thread: record callback(data write)
 * helper thread: ui(keystroke detection)
 */
int main(int argc, char* argv[])
{
	/* ros */
	ros::init(argc, argv, "ros_audio");
	ros::NodeHandle nh;


	/* publish the wake up angle */
	ros::Publisher angle_pub = nh.advertise<std_msgs::Int16>("/Shengyuan", 1000);
	std_msgs::Int16 angle_info;

	/* publish communication cmd */
	std_msgs::Bool communicate_mode;

	/* subscribe microphone command to change the state of 6MIC ARRAY */

	ros::AsyncSpinner spinner(4);
	spinner.start();

init:	
	/*set Serial Port values*/
	serial::Serial ser;
    try
    {
		/* Port need to be changed accordin to system settings */
        ser.setPort("/dev/ttyUSB0");
        ser.setBaudrate(115200);
        serial::Timeout time_out = serial::Timeout::simpleTimeout(1000);
        ser.setTimeout(time_out);
        ser.setStopbits(serial::stopbits_one);
        ser.setFlowcontrol(serial::flowcontrol_none);
        ser.setParity(serial::parity_none);
        ser.open();
    }
    catch (serial::IOException& e)
    {   
        ROS_ERROR_STREAM("Unable to open Serial Port");
		return -1;
    }

    /* make sure the Serial Port is opened */ 
    if(ser.isOpen()) 
    { 
        ROS_INFO_STREAM("Serial Port initialized"); 
    } 
    else 
    { 
        goto init;
    }

	/* the angle when waked up */
	int angle;

	/* SDK from IFlytek*/
	//int ret = MSP_SUCCESS;
	int upload_on =	1; /* whether upload the user word */
	/* login params, please do keep the appid correct */
	const char* login_params = "appid = bf5c5023, work_dir = .";
	int aud_src = 0; /* from mic or file */
	int next = 0;
	bool activated = false;

	/*
	* See "iFlytek MSC Reference Manual"
	*/
	/* Login first. the 1st arg is username, the 2nd arg is password
	 * just set them as NULL. the 3rd arg is login paramertes 
	 * */

	cout << "If you want to use the microphone, please activate it first or please use the communicate mode ..!" << endl;
	while(ros::ok())
	{	
		aud_src = 1;
		if(aud_src != 0) 
		{
			

activate:
			/* process the activate mode here */
			if(microphone_mode != eMicPhone_Communicate)
			{
				/* wake up and localization mode */
				ser.write(LOCALIZATION);
				ser.write(WAKE_UP_YES);
				/* read Serial data */
				if(ser.available())
				{
					/* if we get the wake-up angle, we know the microphone is activated */
					std_msgs::String result;
					result.data = ser.read(ser.available());
					string s = result.data;
					/* get the angle */
					int pos = s.find("angle:");
					if(pos != -1 )
					{	
						cout << "    activated!    " << endl;
						/* activation mode */
						microphone_mode = eMicPhone_Activate;
						activated = true;		
						/* get the angle */
						string ang;
						ang  = s.substr(pos+6,3);
						stringstream ss;
						ss << ang;
						ss >> angle;
						cout << "angle=" << angle << endl; 
						angle_info.data = angle;
						angle_pub.publish(angle_info);
					}
					else
					{
						if(microphone_mode != eMicPhone_Activate)
						{
							//ROS_INFO_STREAM("Please activate the microphone !" << endl);	
							goto activate;
						}
					}
				}
				/* if the microphone is activated, we then get the audio result recognized by SDK and do what we want */
				
			
			}
		}
	}
exit:
	//MSPLogout(); // Logout...

	return 0;
}
