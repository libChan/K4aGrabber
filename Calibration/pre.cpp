#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>           //用于体素网格化的滤波类头文件 
#include <pcl/filters/filter.h>             //滤波相关头文件
#include <pcl/filters/statistical_outlier_removal.h> //统计方法去除离群点
#include <pcl/filters/radius_outlier_removal.h> //统计方法去除离群点
#include <pcl/filters/approximate_voxel_grid.h>  //ApproximateVoxelGrid 

using namespace std;
using namespace pcl;

typedef pcl::PointXYZRGB PointType;

void DownSample(pcl::PointCloud<PointType>::Ptr& cloud_in, pcl::PointCloud<PointType>::Ptr& cloud_out)
{
	//down sample
	std::cout << "begin downSample cloud_in size: " << cloud_in->size() << std::endl;
	pcl::VoxelGrid<PointType> downSampled;  //创建滤波对象
	downSampled.setInputCloud(cloud_in);            //设置需要过滤的点云给滤波对象
	downSampled.setLeafSize(0.01f, 0.01f, 0.01f);  //设置滤波时创建的体素体积为1cm的立方体（1为米，0.01就是1cm）
	downSampled.filter(*cloud_out);  //执行滤波处理，存储输出

	std::cout << "success downSample, size: " << cloud_out->size() << std::endl;

}

void OutlierFilter(pcl::PointCloud<PointType>::Ptr& cloud_in, pcl::PointCloud<PointType>::Ptr& cloud_out)
{
	std::cout << "begin outlierFilter cloud_in size: " << cloud_in->size() << std::endl;

	pcl::RadiusOutlierRemoval<PointType> pcFilter;  //创建滤波器对象
	pcFilter.setInputCloud(cloud_in);             //设置待滤波的点云
	pcFilter.setRadiusSearch(0.03);               // 设置搜索半径
	pcFilter.setMinNeighborsInRadius(3);      // 设置一个内点最少的邻居数目
	pcFilter.filter(*cloud_out);        //滤波结果存储到cloud_filtered

	std::cout << "success OutlierFilter, size: " << cloud_out->size() << std::endl;

}

int main(int argc, char** argv)
{
	//file info
	string master_in = "three_merge.pcd";
	string master_out = "three_merge_pre.pcd";
	string sub_in = "sub2.pcd";
	string sub_out = "sub2_pre.pcd";

	//point cloud
	pcl::PointCloud<PointType>::Ptr cloud_master(new PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr cloud_sub(new PointCloud<PointType>);

	//read point cloud
	pcl::PCDReader reader;
	cout << "reading " << master_in << endl;
	reader.read<PointType>(master_in, *cloud_master);
	//cout << "reading " << sub_in << endl;
	//reader.read<PointType>(sub_in, *cloud_sub);

	//down sample first
	DownSample(cloud_master, cloud_master);
	//DownSample(cloud_sub, cloud_sub);

	//filter outliers
	OutlierFilter(cloud_master, cloud_master);
	//OutlierFilter(cloud_sub, cloud_sub);

	//pcd write
	pcl::PCDWriter writer;
	cout << "writing " << master_out << endl;
	writer.write<PointType>(master_out, *cloud_master);
	//cout << "writing " << sub_out << endl;
	//writer.write<PointType>(sub_out, *cloud_sub);

	return 0;
}