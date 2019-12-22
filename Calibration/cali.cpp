#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>           //用于体素网格化的滤波类头文件 
#include <pcl/filters/filter.h>             //滤波相关头文件
#include <pcl/filters/statistical_outlier_removal.h> //统计方法去除离群点
#include <pcl/filters/radius_outlier_removal.h> //统计方法去除离群点
#include <pcl/filters/approximate_voxel_grid.h>  //ApproximateVoxelGrid 

#include <Open3D/Geometry/PointCloud.h>
#include <Open3d/Registration/ColoredICP.h>
#include <Open3D/Open3D.h>

#include "k4a_grabber.h"

using namespace std;
using namespace boost;
using namespace pcl;
using namespace Eigen;

typedef pcl::PointXYZRGB PointType;
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

template<typename DataType>
Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> load_csv(const std::string& path)
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

template<typename DataType>
void writeToCSVfile(std::string name, Eigen::Array<DataType, -1, -1> matrix)
{
	std::ofstream file(name.c_str());
	file << matrix.format(CSVFormat);
}

void convertToOpenCDPointCloud(pcl::PointCloud<PointType>::ConstPtr cloud, open3d::geometry::PointCloud& pointCloud)
{
	double x, y, z, r, g, b;
	pointCloud.Clear();
	for (const auto& point : cloud->points)
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

Eigen::Matrix4d colorCloudRegistrationOptimization(pcl::PointCloud<PointType>::ConstPtr pcl_source, pcl::PointCloud<PointType>::ConstPtr pcl_target, const Eigen::Matrix4d& init_transformation)
{
	cout << "Start optimization..." << endl;
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

	double radius = 0.3;

	open3d_source_down = open3d_source_current;
	open3d_target_down = open3d_target_current;

	open3d_source_down.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(radius * 2, 30));
	open3d_target_down.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(radius * 2, 30));

	result = open3d::registration::RegistrationColoredICP(open3d_source_down, open3d_target_down,
		radius, current_transformation,
		open3d::registration::ICPConvergenceCriteria(1e-16, 1e-16, 20));
	cout << "rmse: " << result.inlier_rmse_ << endl; 
	return result.transformation_;
}

void getXYZ(const pcl::PointCloud<PointType>& cloud_in, pcl::PointCloud<pcl::PointXYZ>& cloud_out)
{
	for (const auto& point : cloud_in.points)
	{
		pcl::PointXYZ new_point;
		new_point.x = point.x;
		new_point.y = point.y;
		new_point.z = point.z;
		cloud_out.push_back(new_point);
	}

}

int main(int argc, char** argv)
{
	string frame_sub_master_file = "frame_sub2_master.csv";
	cout << "Reading frame_sub_master.csv" << endl;
	Transform<double, 3, Affine> transformation_sub_master;
	transformation_sub_master.matrix() = load_csv<double>(frame_sub_master_file);

	pcl::PCDReader reader;
	// PCL Visualizer
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	Eigen::Matrix4d init_transformation;
	init_transformation = transformation_sub_master.matrix();
	
	Eigen::Matrix4d refinedTransformation;

	// Point Cloud
	pcl::PointCloud<PointType>::Ptr cloud_master(new PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr cloud_sub(new PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr cloud_sub_temp(new PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr cloud(new PointCloud<PointType>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sub_xyz(new PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_master_xyz(new PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new PointCloud<pcl::PointXYZ>);


	reader.read<PointType>("master2_pre.pcd", *cloud_master);
	cout << "read master.pcd..." << endl;
	reader.read<PointType>("sub2_pre.pcd", *cloud_sub);
	cout << "read sub.pcd..." << endl;

	/*
	//查看初始标定效果
	transformPointCloud(*cloud_sub, *cloud_sub, init_transformation, true);
	*cloud = *cloud_master + *cloud_sub;
	viewer->addPointCloud<PointType>(cloud, "cloud");
	viewer->spinOnce(600000);
	
	
	transformPointCloud(*cloud_sub, *cloud_sub_temp, init_transformation, true); //初始标定矩阵先转化点云
	
	//转换为XYZ格式点云
	getXYZ(*cloud_master, *cloud_master_xyz);
	getXYZ(*cloud_sub_temp, *cloud_sub_xyz);

	cout<<"begin ICP..."<<endl;
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setMaxCorrespondenceDistance(0.5);
	icp.setMaximumIterations(20);
	icp.setTransformationEpsilon(1e-8);
	icp.setEuclideanFitnessEpsilon(1);
	icp.setInputCloud(cloud_sub_xyz);
	icp.setInputTarget(cloud_master_xyz);
	icp.align(*cloud_out);
	cout << "score: " << icp.getFitnessScore() << endl;

	//打印优化得到的转化矩阵
	cout << "icp优化转换矩阵：" << endl;
	cout << icp.getFinalTransformation() << endl;

	transformPointCloud(*cloud_sub, *cloud_sub, icp.getFinalTransformation(), true); //优化矩阵先转化点云
	*cloud = *cloud_master + *cloud_sub;
	viewer->addPointCloud(cloud, "cloud");
	viewer->spinOnce(600000);
	
	pcl::PCDWriter writer;
	writer.write<PointType>("sub2_down.pcd", *cloud_sub, false);
	cout << "sub_down.pcd save ok" << endl;
	writer.write<PointType>("master2_down.pcd", *cloud_master, false);
	cout << "master_down.pcd save ok" << endl;
	*/

	refinedTransformation = colorCloudRegistrationOptimization(cloud_sub, cloud_master, init_transformation);
	cout << "refined matrix:" << endl;
	cout << refinedTransformation.matrix() << endl;
	writeToCSVfile<double>("matrix_optimal2.csv", refinedTransformation.matrix());
	transformPointCloud(*cloud_sub, *cloud_sub, refinedTransformation, true);
	*cloud = *cloud_master + *cloud_sub;

	viewer->removePointCloud("cloud");
	viewer->addPointCloud<PointType>(cloud, "cloud");
	viewer->spinOnce(600000);

	return 0;
}