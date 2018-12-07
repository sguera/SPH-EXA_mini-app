#pragma once

class Scheduler
{

	public:
	
		add(Task t){
			tasks.push_back(t);
		}

		exec(){
			for (auto i : tasks)
			{
				i.exec();
			}
		}

	private:

		std::vector<Task> tasks;
}