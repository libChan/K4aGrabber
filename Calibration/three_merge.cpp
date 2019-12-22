#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

using namespace std;
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

int main(int argc, char** argv)
{
	string master = "master1_pre.pcd";
	string sub1 = "sub1_pre.pcd";
	string sub2 = "sub2_pre.pcd";

	//point cloud
	pcl::PointCloud<PointType>::Ptr cloud(new PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr cloud_master(new PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr cloud_sub1(new PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr cloud_sub2(new PointCloud<PointType>);
	
	//read point cloud
	pcl::PCDReader reader;
	cout << "reading cloud master" << endl;
	reader.read<PointType>(master, *cloud_master);
	cout << "reading cloud sub1" << endl;
	reader.read<PointType>(sub1, *cloud_sub1);
	cout << "reading cloud sub2" << endl;
	reader.read<PointType>(sub2, *cloud_sub2);

	//read csv
	cout << "loading csv" << endl;
	Eigen::Matrix4d init_transformation_sub1;
	Eigen::Matrix4d init_transformation_sub2;
	init_transformation_sub1 = load_csv<double>("matrix_optimal1.csv");
	init_transformation_sub2 = load_csv<double>("matrix_optimal2.csv");

	//transform point cloud
	transformPointCloud(*cloud_sub1, *cloud_sub1, init_transformation_sub1, true);
	transformPointCloud(*cloud_sub2, *cloud_sub2, init_transformation_sub2, true);

	*cloud = *cloud_master + *cloud_sub1;
	*cloud += *cloud_sub2;
	//save merge cloud point;
	pcl::PCDWriter writer;
	cout << "saving merge cloud..." << endl;
	writer.write<PointType>("three_merge.pcd", *cloud);

	// PCL Visualizer
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->addPointCloud(cloud, "cloud");
	viewer->spinOnce(6000000);

	return 0;

}