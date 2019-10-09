#include "BBox.hpp"
#include <vector>
#include <future>

void gnuplot_init(FILE *&gp)
{
    gp = popen("gnuplot -persistent", "w");
    fprintf(gp, "set pointsize 1;\n");
    fprintf(gp, "set cbrange [0:25];\n");
    //fprintf(gp, "set pointtype 6;\n");
    //fprintf(gp, "set pm3d map;\n");
    //fprintf(gp, "set pm3d interpolate 2,2\n");
    //fprintf(gp, "set terminal wxt size 600,600\n");
    //fprintf(gp, "set cbrange [300.0:700.0];\n");
    //fprintf(gp, "set palette defined ( 0 \"black\", 1 \"red\", 2 \"yellow\", 3 \"white\" );\n");
}

void gnuplot_destroy(FILE *gp)
{
    fprintf(gp, "exit\n");
    pclose(gp);
}

static std::vector<float> xy;
static std::future<void> future_plot;

template <typename T, class Dataset>
void gnuplot_plot(FILE *gp, const std::vector<int> &clist, const Dataset &d)
{
    if(d.iteration % 1 == 0)
    {
        if(d.iteration > 1)
        {
            future_plot.wait();
            future_plot.get();
        }

        const int64_t n = clist.size();

        const T *x = d.x.data();
        const T *y = d.y.data();
        const T *ro = d.ro.data();
        //const int *neighborsCount = d.neighborsCount.data();

        int step = 1;
        int np = n / step;

        if((int)xy.size() < np) xy.resize(np*3);

        #pragma omp parallel for
        for(int pi=0; pi<np; pi++)
        {   
            int i = clist[pi*step];

            xy[pi*3] = static_cast<float>(x[i]);
            xy[pi*3+1] = static_cast<float>(y[i]);
            xy[pi*3+2] = static_cast<float>(ro[i]);
            //xy[pi*3+2] = neighborsCount[pi];
        }

        float xmin = d.bbox.xmin;
        float xmax = d.bbox.xmax;
        float ymin = d.bbox.ymin;
        float ymax = d.bbox.ymax;

        future_plot = std::async(std::launch::async, [gp,np,xmin,xmax,ymin,ymax](){
            fprintf(gp, "set xrange [%f:%f];\n", xmin, xmax);
            fprintf(gp, "set yrange [%f:%f];\n", ymin, ymax);
            //fprintf(gp, "set cbrange [-50.0:50.0];\n");
            // pt 7 ps 1 
            fprintf(gp, "plot '-' binary format='%%float%%float%%float' record=%d u 1:2:3 w p pt 7 lc palette z\n", np);
            fwrite(xy.data(), sizeof(float), 3*np, gp);
            fprintf(gp, "\n");
            fflush(gp);
        });
    }
}

void gnuplot_wait()
{
    // future_plot.wait();
    // future_plot.get();
}