#include <vector>
#include <iterator>

#include "point3d.h"
#include "boundingbox.h"
#include "octree.h"

using std::vector;
using std::array;

template <typename T>
struct ValuePoint {
  Point3d dimensions_;
  T value_;
};

template <typename T>
bool operator==(const ValuePoint<T>& lhs, const ValuePoint<T>& rhs) {
  return lhs.dimensions_ == rhs.dimensions_ && lhs.value_ == rhs.value_;
}

template <typename T>
struct ExamplePointExtractor {
  Point3d operator()(const ValuePoint<T>& p) {
    return p.dimensions_;
  }
};

int main(int argc, char** argv)
{
	double random_x, random_y, random_z;
	BoundingBox allBox;
	std::vector<ValuePoint<int>> data(100);

	for (double i = 0; i < data.size(); ++i) {
		random_x = std::rand()/((double)RAND_MAX+1);
		random_y = std::rand()/((double)RAND_MAX+1);
		random_z = std::rand()/((double)RAND_MAX+1);

		data[i].dimensions_ = Point3d{ random_x, random_y, random_z };
	    data[i].value_ = static_cast<int>(i);
		allBox.maxes_.x = allBox.maxes_.y = allBox.maxes_.z = i + 2;
	}
	Octree<std::vector<ValuePoint<int>>::iterator, ExamplePointExtractor<int>> o(data.begin(), data.end());
	allBox.mins_.x = allBox.mins_.y = allBox.mins_.z = 0;


    vector<vector<ValuePoint<int>>::const_iterator> expectedValues;
    for (auto it = data.begin(); it != data.end(); ++it) {
        if (allBox.contains(ExamplePointExtractor<int>()(*it))) {
            expectedValues.push_back(it);
        }
    }

    // for (size_t index = 0; index < expectedValues.size(); ++index)
    // 		std::cout << "at index: " << index << "\n";

    BoundingBox box{{-5, -5, -5}, {-10, -10, -10}};
    vector<vector<ValuePoint<int>>::const_iterator> outputValues;
    auto outputIterator = back_inserter(outputValues);
    if(false == o.search(box, outputIterator)) std::cout << "Not present\n";

    BoundingBox extrema{{0, 0, 0}, {100, 100, 100}};
	array<BoundingBox, 8> partitions = extrema.partition();
	array<BoundingBox, 8> expectedPartitions{{
		BoundingBox{{0, 0, 0}, {50, 50, 50}},
		BoundingBox{{50, 0, 0}, {100, 50, 50}},
		BoundingBox{{0, 50, 0}, {50, 100, 50}},
		BoundingBox{{50, 50, 0}, {100, 100, 50}},
		BoundingBox{{0, 0, 50}, {50, 50, 100}},
		BoundingBox{{50, 0, 50}, {100, 50, 100}},
		BoundingBox{{0, 50, 50}, {50, 100, 100}},
		BoundingBox{{50, 50, 50}, {100, 100, 100}}
	}};
	for (unsigned i = 0; i < 8; ++i) {
		if(partitions[i]== expectedPartitions[i]) std::cout << "Correctly partitioned\n";
	}

	for(auto valuepoint: o){
		std::cout << valuepoint.dimensions_ << " is in " <<  << "\n"
	}

}


