#ifndef _MODEL_HPP
#define _MODEL_HPP

#include <vector>
#include <string>
#include <iostream>


class Model
{
public:

	Model();
    ~Model();

    void iterate();

private:
    TimeScheme time_s;
    std::vector<Physics> physics;

};

#endif
