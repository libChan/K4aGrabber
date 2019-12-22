#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>           //�����������񻯵��˲���ͷ�ļ� 
#include <pcl/filters/filter.h>             //�˲����ͷ�ļ�
#include <pcl/filters/statistical_outlier_removal.h> //ͳ�Ʒ���ȥ����Ⱥ��
#include <pcl/filters/radius_outlier_removal.h> //ͳ�Ʒ���ȥ����Ⱥ��
#include <pcl/filters/approximate_voxel_grid.h>  //ApproximateVoxelGrid 

using namespace std;
using namespace pcl;

typedef pcl::PointXYZRGB PointType;

void DownSample(pcl::PointCloud<PointType>::Ptr& cloud_in, pcl::PointCloud<PointType>::Ptr& cloud_out)
{
	//down sample
	std::cout << "begin downSample cloud_in size: " << cloud_in->size() << std::endl;
	pcl::VoxelGrid<PointType> downSampled;  //�����˲�����
	downSampled.setInputCloud(cloud_in);            //������Ҫ���˵ĵ��Ƹ��˲�����
	downSampled.setLeafSize(0.01f, 0.01f, 0.01f);  //�����˲�ʱ�������������Ϊ1cm�������壨1Ϊ�ף�0.01����1cm��
	downSampled.filter(*cloud_out);  //ִ���˲������洢���

	std::cout << "success downSample, size: " << cloud_out->size() << std::endl;

}

void OutlierFilter(pcl::PointCloud<PointType>::Ptr& cloud_in, pcl::PointCloud<PointType>::Ptr& cloud_out)
{
	std::cout << "begin outlierFilter cloud_in size: " << cloud_in->size() << std::endl;

	pcl::RadiusOutlierRemoval<PointType> pcFilter;  //�����˲�������
	pcFilter.setInputCloud(cloud_in);             //���ô��˲��ĵ���
	pcFilter.setRadiusSearch(0.03);               // ���������뾶
	pcFilter.setMinNeighborsInRadius(3);      // ����һ���ڵ����ٵ��ھ���Ŀ
	pcFilter.filter(*cloud_out);        //�˲�����洢��cloud_filtered

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