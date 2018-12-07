#pragma once

class ComputeParticleTask : public Task
{

	public:

		ComputeParticleTask(int size){
			size = size;
		}
		
		virtual preprocess(){}

		virtual compute(int particle_id) = 0;

		virtual postprocesss() {}

		void exec(){

			preprocess();

			#pragma omp parallel for
			for (i = 0; i < size; ++i) {
				compute(i);
			}

			postprocess();
			
		}

	private:
		int size;

}
