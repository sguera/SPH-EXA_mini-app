#pragma once

namespace sphexa
{
template <typename Dataset>
class Printer
{
public:
    Printer(const Dataset &d)
        : d(d)
    {
    }

    void printConstants(const int iteration, const int nntot, std::ostream &out)
    {
        if (d.rank == 0)
        {
            out << iteration << ' ' << d.ttot << ' ' << d.minDt << ' ' << d.etot << ' ' << d.ecin << ' ' << d.eint << ' ' << nntot << ' '
                << std::endl;
            out.flush();
        }
    }

    void printCheck(const size_t particleCount, const size_t nodeCount, const size_t haloCount, const size_t totalNeighbors,
                    std::ostream &out)
    {
        out << "### Check ### Global Tree Nodes: " << nodeCount << ", Particles: " << particleCount << ", Halos: " << haloCount
            << std::endl;
        out << "### Check ### Computational domain: " << d.bbox.xmin << " " << d.bbox.xmax << " " << d.bbox.ymin << " " << d.bbox.ymax
            << " " << d.bbox.zmin << " " << d.bbox.zmax << std::endl;
        out << "### Check ### Total Neighbors: " << totalNeighbors << ", Avg neighbor count per particle: " << totalNeighbors / d.n
            << std::endl;
        out << "### Check ### Total time: " << d.ttot << ", current time-step: " << d.minDt << std::endl;
        out << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", cinetic: " << d.ecin << ")" << std::endl;
    }

    void printAllDataToFile(const std::vector<int> &clist, const std::string &dumpfilename)
    {
        for (int turn = 0; turn < d.nrank; turn++)
        {
            if (turn == d.rank)
            {
                std::ofstream dump;

                if (d.rank == 0)
                    dump.open(dumpfilename);
                else
                    dump.open(dumpfilename, std::ios_base::app);

                for (unsigned int pi = 0; pi < clist.size(); pi++)
                {
                    const int i = clist[pi];
                    // const int nn = d.neighborsCount[pi];

                    dump << d.x[i] << ' ' << d.y[i] << ' ' << d.z[i] << ' ';
                    dump << d.vx[i] << ' ' << d.vy[i] << ' ' << d.vz[i] << ' ';
                    dump << d.h[i] << ' ' << d.ro[i] << ' ' << d.u[i] << ' ' << d.p[i] << ' ' << d.c[i] << ' ';
                    dump << d.grad_P_x[i] << ' ' << d.grad_P_y[i] << ' ' << d.grad_P_z[i] << ' ';
                    // T rad = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
                    // T vrad = (vx[i] * x[i] + vy[i] * y[i] + vz[i] * z[i]) / rad;
                    // dump << rad << ' ' << vrad << std::endl;
                    dump << d.rank << std::endl;
                }

                dump.close();

#ifdef USE_MPI
                MPI_Barrier(MPI_COMM_WORLD);
#endif
            }
            else
            {
#ifdef USE_MPI
                MPI_Barrier(MPI_COMM_WORLD);
#endif
            }
        }
    }

    void printTotalIterationTime(const float duration, std::ostream &out)
    {
        out << "=== Total time for iteration(" << d.iteration << ") " << duration << "s" << std::endl << std::endl;
    }

private:
    const Dataset &d;
}; // namespace sphexa
} // namespace sphexa
