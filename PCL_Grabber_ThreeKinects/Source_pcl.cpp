#include <k4a/k4a.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <array>

#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Open3D/Geometry/PointCloud.h>
#include <Open3d/Registration/ColoredICP.h>
#include <Open3D/Open3D.h>
#include "k4a_grabber.h"


using namespace std;
using namespace boost;
using namespace pcl;
using namespace Eigen;
using namespace glm;

typedef pcl::PointXYZRGB PointType;

template<typename DataType>
Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> load_csv(const std::string & path)
{
	std::ifstream indata;
	indata.open(path);
	std::string line;
	std::vector<DataType> values;
	unsigned int rows = 0;
	while (std::getline(indata, line))
	{
		std::stringstream lineStream(line);
		std::string cell;
		while (std::getline(lineStream, cell, ','))
		{
			values.push_back(std::stod(cell));
		}
		++rows;
	}
	return Eigen::Map<const Eigen::Matrix<typename Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>::Scalar, Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>::RowsAtCompileTime, Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size() / rows);
}

void convertToOpenCDPointCloud(pcl::PointCloud<PointType>::ConstPtr cloud, open3d::geometry::PointCloud &pointCloud)
{
	double x, y, z, r, g, b;
	pointCloud.Clear();

	for (const auto &point : cloud->points)
	{
		x = point.x;
		y = point.y;
		z = point.z;
		r = point.r / 256.0;
		g = point.g / 256.0;
		b = point.b / 256.0;

		pointCloud.points_.push_back(Eigen::Vector3d(x, y, z));
		pointCloud.colors_.push_back(Eigen::Vector3d(r, g, b));
	}
}

Eigen::Matrix4d colorCloudRegistrationOptimization(pcl::PointCloud<PointType>::ConstPtr pcl_source, pcl::PointCloud<PointType>::ConstPtr pcl_target, const Eigen::Matrix4d &init_transformation)
{
	open3d::geometry::PointCloud open3d_source_original, open3d_target_original;
	open3d::geometry::PointCloud open3d_source_current, open3d_target_current;

	convertToOpenCDPointCloud(pcl_source, open3d_source_original);
	convertToOpenCDPointCloud(pcl_target, open3d_target_original);

	open3d::registration::RegistrationResult result;
	Eigen::Matrix4d current_transformation = init_transformation;

	open3d_source_current = open3d_source_original;
	open3d_target_current = open3d_target_original;

	open3d::geometry::PointCloud open3d_source_down;
	open3d::geometry::PointCloud open3d_target_down;

	double radius = 0.01;

	open3d_source_down = *open3d_source_current.VoxelDownSample(radius);
	open3d_target_down = *open3d_target_current.VoxelDownSample(radius);

	open3d_source_down.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(radius * 2, 30));
	open3d_target_down.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(radius * 2, 30));

	result = open3d::registration::RegistrationColoredICP(open3d_source_down, open3d_target_down,
		radius, current_transformation,
		open3d::registration::ICPConvergenceCriteria(1e-6, 1e-6, 50));

	return result.transformation_;
}

int main(int argc, char **argv)
{
	string frame_sub1_master_file = "frame_sub1_master.csv";
	string frame_sub2_master_file = "frame_sub2_master.csv";
	cout << "Reading csv file" << endl;
	Transform<double, 3, Affine> transformation_sub1_master;
	Transform<double, 3, Affine> transformation_sub2_master;
	transformation_sub1_master.matrix() = load_csv<double>(frame_sub1_master_file);
	transformation_sub2_master.matrix() = load_csv<double>(frame_sub2_master_file);

	const uint32_t deviceCount = k4a::device::get_installed_count();
	cout << deviceCount << " devices connected..." << endl;
	if (deviceCount == 0)
	{
		cout << "no azure kinect devices detected!" << endl;
	}
	if (deviceCount != 3)
	{
		cout << "should connect 3 kinects" << endl;
	}

	// PCL Visualizer
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));

	// Point Cloud
	pcl::PointCloud<PointType>::Ptr cloud_master;
	pcl::PointCloud<PointType>::Ptr cloud_sub1;
	pcl::PointCloud<PointType>::Ptr cloud_sub2;


	// Retrieved Point Cloud Callback Function
	boost::mutex mutex_master;
	boost::function<void(const pcl::PointCloud<PointType>::ConstPtr&)> function_master =
		[&cloud_master, &mutex_master](const pcl::PointCloud<PointType>::ConstPtr& ptr)
	{
		boost::mutex::scoped_lock lock(mutex_master);

		/* Point Cloud Processing */

		cloud_master = ptr->makeShared();

	};

	boost::mutex mutex_sub1;
	boost::function<void(const pcl::PointCloud<PointType>::ConstPtr&)> function_sub1 =
		[&cloud_sub1, &mutex_sub1](const pcl::PointCloud<PointType>::ConstPtr& ptr)
	{
		boost::mutex::scoped_lock lock(mutex_sub1);

		/* Point Cloud Processing */

		cloud_sub1 = ptr->makeShared();

	};

	boost::mutex mutex_sub2;
	boost::function<void(const pcl::PointCloud<PointType>::ConstPtr&)> function_sub2 =
		[&cloud_sub2, &mutex_sub2](const pcl::PointCloud<PointType>::ConstPtr& ptr)
	{
		boost::mutex::scoped_lock lock(mutex_sub2);

		/* Point Cloud Processing */

		cloud_sub2 = ptr->makeShared();

	};


	// KinectAzureDKGrabber
	boost::shared_ptr<pcl::Grabber> grabber_sub2 =
		boost::make_shared<pcl::KinectAzureDKGrabber>(2, K4A_DEPTH_MODE_NFOV_UNBINNED, K4A_IMAGE_FORMAT_COLOR_BGRA32, K4A_COLOR_RESOLUTION_720P);
	boost::shared_ptr<pcl::Grabber> grabber_sub1 =
		boost::make_shared<pcl::KinectAzureDKGrabber>(1, K4A_DEPTH_MODE_NFOV_UNBINNED, K4A_IMAGE_FORMAT_COLOR_BGRA32, K4A_COLOR_RESOLUTION_720P);
	boost::shared_ptr<pcl::Grabber> grabber_master =
		boost::make_shared<pcl::KinectAzureDKGrabber>(0, K4A_DEPTH_MODE_NFOV_UNBINNED, K4A_IMAGE_FORMAT_COLOR_BGRA32, K4A_COLOR_RESOLUTION_720P);

	boost::shared_ptr<pcl::KinectAzureDKGrabber> grabber_ = boost::dynamic_pointer_cast<pcl::KinectAzureDKGrabber>(grabber_master);

	// Register Callback Function
	boost::signals2::connection connection_sub1 = grabber_sub1->registerCallback(function_sub1);
	boost::signals2::connection connection_sub2 = grabber_sub2->registerCallback(function_sub2);
	boost::signals2::connection connection_master = grabber_master->registerCallback(function_master);

	// Start Grabber
	grabber_sub2->start();
	grabber_sub1->start();
	grabber_master->start();

	// get the intrinics of master
	k4a::calibration calibration = grabber_->getCalibration();
	k4a_calibration_intrinsic_parameters_t *intrinsics = &calibration.color_camera_calibration.intrinsics.parameters;
	Eigen::Matrix3f intrinsics_eigen;
	intrinsics_eigen <<
		intrinsics->param.fx, 0.0f, intrinsics->param.cx,
		0.0f, intrinsics->param.fy, intrinsics->param.cy,
		0.0f, 0.0f, 1.0f;
	Eigen::Matrix4f extrinsics_eigen = Eigen::Matrix4f::Identity();
	viewer->setCameraParameters(intrinsics_eigen, extrinsics_eigen);

	pcl::PointCloud<PointType>::Ptr cloud(new PointCloud<PointType>);
	float aspect = 0.475f;
	
	transformation_sub1_master.matrix()(0, 3) *= aspect;
	transformation_sub1_master.matrix()(1, 3) *= aspect;
	transformation_sub1_master.matrix()(2, 3) *= aspect;
	transformation_sub2_master.matrix()(0, 3) *= aspect;
	transformation_sub2_master.matrix()(1, 3) *= aspect;
	transformation_sub2_master.matrix()(2, 3) *= aspect;

	bool refineResult = false;

	Eigen::Matrix4d init_transformation1;
	Eigen::Matrix4d init_transformation2;
	init_transformation1 = transformation_sub1_master.matrix();
	init_transformation2 = transformation_sub2_master.matrix();

	Eigen::Matrix4d refinedTransformation1;
	Eigen::Matrix4d refinedTransformation2;
	

	while (!viewer->wasStopped())
	{
		// Update Viewer
		viewer->spinOnce();
		boost::mutex::scoped_try_lock lock_master(mutex_master);
		boost::mutex::scoped_try_lock lock_sub1(mutex_sub1);
		boost::mutex::scoped_try_lock lock_sub2(mutex_sub2);

		if (lock_master.owns_lock() && cloud_master &&
			lock_sub1.owns_lock() && cloud_sub1 &&
			lock_sub2.owns_lock() && cloud_sub2)
		{
			
			if (!refineResult)
			{
				std::cout << "Start Calibration Optimization." << std::endl;
				refinedTransformation1 = colorCloudRegistrationOptimization(cloud_sub1, cloud_master, init_transformation1);
				refinedTransformation2 = colorCloudRegistrationOptimization(cloud_sub2, cloud_master, init_transformation2);
				refineResult = true;
				std::cout << "Calibration optimization process over." << std::endl;
			}

			transformPointCloud(*cloud_sub1, *cloud_sub1, refinedTransformation1, true);
			transformPointCloud(*cloud_sub2, *cloud_sub2, refinedTransformation2, true);

			*cloud = *cloud_sub1 + *cloud_sub2;
			*cloud += *cloud_master;
			

			// Update Point Cloud
			if (!viewer->updatePointCloud(cloud, "cloud"))
			{
				viewer->addPointCloud(cloud, "cloud");
			}
		}
	}
	// Stop Grabber
	grabber_sub1->stop();
	grabber_sub2->stop();
	grabber_master->stop();

	// Disconnect Callback Function
	if (connection_master.connected() || connection_sub1.connected() || connection_sub2.connected())
	{
		connection_sub1.disconnect();
		connection_sub2.disconnect();
		connection_master.disconnect();
	}

	return 0;
}