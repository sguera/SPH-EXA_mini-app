#include <iostream>

#include "ParticlesData.hpp"
#include "IFileReader.hpp"

namespace sphexa
{

template <typename Dataset>
struct EvrardCollapseInputFileReader : IFileReader<Dataset>
{
    Dataset readParticleDataFromBinFile(const std::string &path, const size_t noParticles) const override
    {
        Dataset pd;
        pd.n = noParticles;
        pd.resize(noParticles);
        pd.count = pd.x.size();

        try
        {
            printf("Loading input file with %lu particles for Evrard Collapse at path '%s'... ", pd.n, path.c_str());
            fileutils::readParticleDataFromBinFile(path, pd.x, pd.y, pd.z, pd.vx, pd.vy, pd.vz, pd.ro, pd.u, pd.p, pd.h, pd.m);
            printf("OK\n");

            init(pd);
        }
        catch (FileNotOpenedException &ex)
        {
            printf("ERROR: %s. Terminating\n", ex.what());
            exit(EXIT_FAILURE);
        }

        return pd;
    }

    Dataset readParticleDataFromCheckpointBinFile(const std::string &path) const override
    {
        Dataset pd;
        std::ifstream inputfile(path, std::ios::binary);

        if (inputfile.is_open())
        {
            inputfile.read(reinterpret_cast<char *>(&pd.n), sizeof(size_t));

            pd.resize(pd.n);

            pd.n = pd.x.size();
            pd.count = pd.x.size();

            printf("Loading checkpoint file with %lu particles for Evrard Collapse... ", pd.n);

            inputfile.read(reinterpret_cast<char *>(&pd.ttot), sizeof(pd.ttot));
            inputfile.read(reinterpret_cast<char *>(&pd.minDt), sizeof(pd.minDt));

            fileutils::details::readParticleDataFromBinFile(inputfile, pd.x, pd.y, pd.z, pd.vx, pd.vy, pd.vz, pd.ro, pd.u, pd.p, pd.h, pd.m,
                                                            pd.temp, pd.mue, pd.mui, pd.du, pd.du_m1, pd.dt, pd.dt_m1, pd.x_m1, pd.y_m1,
                                                            pd.z_m1);
            inputfile.close();

            std::fill(pd.grad_P_x.begin(), pd.grad_P_x.end(), 0.0);
            std::fill(pd.grad_P_y.begin(), pd.grad_P_y.end(), 0.0);
            std::fill(pd.grad_P_z.begin(), pd.grad_P_z.end(), 0.0);

            std::fill(pd.fx.begin(), pd.fx.end(), 0);
            std::fill(pd.fy.begin(), pd.fy.end(), 0);
            std::fill(pd.fz.begin(), pd.fz.end(), 0);
            std::fill(pd.ugrav.begin(), pd.ugrav.end(), 0);

            pd.etot = pd.ecin = pd.eint = pd.egrav = 0.0;

            printf("OK\n");
        }
        else
            printf("ERROR: Can't open file %s\n", path.c_str());
        return pd;
    }

protected:
    void init(Dataset &pd) const
    {
        // additional fields for evrard
        std::fill(pd.temp.begin(), pd.temp.end(), 1.0);
        std::fill(pd.mue.begin(), pd.mue.end(), 2.0);
        std::fill(pd.mui.begin(), pd.mui.end(), 10.0);
        std::fill(pd.vx.begin(), pd.vx.end(), 0.0);
        std::fill(pd.vy.begin(), pd.vy.end(), 0.0);
        std::fill(pd.vz.begin(), pd.vz.end(), 0.0);
        std::fill(pd.grad_P_x.begin(), pd.grad_P_x.end(), 0.0);
        std::fill(pd.grad_P_y.begin(), pd.grad_P_y.end(), 0.0);
        std::fill(pd.grad_P_z.begin(), pd.grad_P_z.end(), 0.0);
        std::fill(pd.du.begin(), pd.du.end(), 0.0);
        std::fill(pd.du_m1.begin(), pd.du_m1.end(), 0.0);
        std::fill(pd.dt.begin(), pd.dt.end(), 0.0001);
        std::fill(pd.dt_m1.begin(), pd.dt_m1.end(), 0.0001);

        std::fill(pd.fx.begin(), pd.fx.end(), 0);
        std::fill(pd.fy.begin(), pd.fy.end(), 0);
        std::fill(pd.fz.begin(), pd.fz.end(), 0);
        std::fill(pd.ugrav.begin(), pd.ugrav.end(), 0);

        for (size_t i = 0; i < pd.count; ++i)
        {
            pd.x_m1[i] = pd.x[i] - pd.vx[i] * pd.dt[0];
            pd.y_m1[i] = pd.y[i] - pd.vy[i] * pd.dt[0];
            pd.z_m1[i] = pd.z[i] - pd.vz[i] * pd.dt[0];
        }
        pd.etot = pd.ecin = pd.eint = pd.egrav = 0.0;
        pd.minDt = 1e-4;
    }
};

#ifdef USE_MPI
template <typename Dataset>
struct EvrardCollapseMPIInputFileReader : EvrardCollapseInputFileReader<Dataset>
{
    Dataset readParticleDataFromBinFile(const std::string &path, const size_t noParticles) const override
    {
        Dataset d;
        d.n = noParticles;
        initMPIData(d);

        try
        {
            fileutils::readParticleDataFromBinFileWithMPI(path, d, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.ro, d.u, d.p, d.h, d.m);
            if (d.rank == 0) printf("Loaded input file with %lu particles for Evrard Collapse from path '%s' \n", d.n, path.c_str());
        }
        catch (MPIFileNotOpenedException &ex)
        {
            if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            MPI_Abort(d.comm, ex.mpierr);
        }

        this->init(d);

        return d;
    }

    Dataset readParticleDataFromCheckpointBinFile(const std::string &path) const override
    {
        Dataset d;
        initMPIData(d);

        try
        {
            fileutils::readParticleCheckpointDataFromBinFileWithMPI(path, d, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.ro, d.u, d.p, d.h, d.m,
                                                                    d.temp, d.mue, d.mui, d.du, d.du_m1, d.dt, d.dt_m1, d.x_m1, d.y_m1,
                                                                    d.z_m1);
            if (d.rank == 0) printf("Loaded checkpoint file with %lu particles for Evrard Collapse from path '%s'\n", d.n, path.c_str());
        }
        catch (MPIFileNotOpenedException &ex)
        {
            if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            MPI_Abort(d.comm, ex.mpierr);
        }

        std::fill(d.grad_P_x.begin(), d.grad_P_x.end(), 0.0);
        std::fill(d.grad_P_y.begin(), d.grad_P_y.end(), 0.0);
        std::fill(d.grad_P_z.begin(), d.grad_P_z.end(), 0.0);

        std::fill(d.fx.begin(), d.fx.end(), 0);
        std::fill(d.fy.begin(), d.fy.end(), 0);
        std::fill(d.fz.begin(), d.fz.end(), 0);
        std::fill(d.ugrav.begin(), d.ugrav.end(), 0);

        d.etot = d.ecin = d.eint = d.egrav = 0.0;

        return d;
    }

private:
    void initMPIData(Dataset &d) const
    {
        d.comm = MPI_COMM_WORLD;
        MPI_Comm_size(d.comm, &d.nrank);
        MPI_Comm_rank(d.comm, &d.rank);
        MPI_Get_processor_name(d.pname, &d.pnamelen);
    }
};
#endif
} // namespace sphexa
