#pragma once

namespace sphexa
{
namespace sph
{

template <typename T>
void treeWalkForRemoteParticlesRef(const Octree<T> &node, const int i, const T *xi, const T *yi, const T *zi, const T *hi, const T *hj,
                                   const T *mj, T *fx, T *fy, T *fz, T *ugrav)
{
    const auto gnode = dynamic_cast<const GravityOctree<T> &>(node);

    if (gnode.particleIdxList.empty()) return; // skip empty nodes

    const T d1 = std::abs(xi[i] - gnode.xce);
    const T d2 = std::abs(yi[i] - gnode.yce);
    const T d3 = std::abs(zi[i] - gnode.zce);
    const T dc = 4.0 * hi[i] + gnode.dx / 2.0;

    if (d1 <= dc && d2 <= dc && d3 <= dc) // intersecting
    {
        if (gnode.dx == 0) // node is a leaf
        {
            // If tree node assignee is -1 it means that this tree node is shared across a few computing nodes.
            // Gravity contribution of that nodes was already calculated during the first gravity (self Gravity) calculations
            // Return below is to avoid calculating it twice
            if (gnode.assignee == -1) return;

            const auto j = gnode.particleIdx;

            // if (i != j) // skip calculating gravity contribution of myself
            if (!(xi[i] == gnode.xce && yi[i] == gnode.yce && zi[i] == gnode.zce))
            {
                const T dd2 = d1 * d1 + d2 * d2 + d3 * d3;
                const T dd5 = std::sqrt(dd2);

                T g0;

                if (dd5 > 2.0 * hi[i] && dd5 > 2.0 * hj[j]) { g0 = 1.0 / dd5 / dd2; }
                else
                {
                    const T hij = hi[i] + hj[j];
                    const T vgr = dd5 / hij;
                    const T mefec = std::min(1.0, vgr * vgr * vgr);
                    g0 = mefec / dd5 / dd2;
                }
                const T r1 = xi[i] - gnode.xcm;
                const T r2 = yi[i] - gnode.ycm;
                const T r3 = zi[i] - gnode.zcm;

                fx[i] -= g0 * r1 * mj[j];
                fy[i] -= g0 * r2 * mj[j];
                fz[i] -= g0 * r3 * mj[j];
                ugrav[i] += g0 * dd2 * mj[j];
#ifndef NDEBUG
                if (std::isnan(fx[i])) printf("i=%d fx[i]=%.15f, g0=%f\n", i, fx[i], g0);
#endif
            }
        }
        else
        {
            for (const auto &child : gnode.cells) // go deeper to the childs
            {
                if (child->global && child->assignee != child->comm_rank && child->assignee != -1) { continue; }

                treeWalkForRemoteParticlesRef(*child, i, xi, yi, zi, hi, hj, mj, fx, fy, fz, ugrav);
            }
        }
    }
    else // not intersecting
    {
        const T r1 = xi[i] - gnode.xcm;
        const T r2 = yi[i] - gnode.ycm;
        const T r3 = zi[i] - gnode.zcm;
        const T dd2 = r1 * r1 + r2 * r2 + r3 * r3;

        if (gnode.dx * gnode.dx <= gravityTolerance * dd2)
        {
            // If tree node assignee is -1 it means that this tree node is shared across a few computing nodes.
            // Gravity contribution of that nodes was already calculated during the first gravity (self Gravity) calculations
            // Return below is to avoid calculating it twice
            if (gnode.assignee == -1) return;

            const T dd5 = sqrt(dd2);
            const T d32 = 1.0 / dd5 / dd2;

            T g0;

            if (gnode.dx == 0) // node is a leaf
            {
                const int j = gnode.particleIdx;
                const T v1 = dd5 / hi[i];
                const T v2 = dd5 / hj[j];

                if (v1 > 2.0 && v2 > 2.0) { g0 = gnode.mTot * d32; }
                else
                {
                    const T hij = hi[i] + hj[j];
                    const T vgr = dd5 / hij;
                    const T mefec = std::min(1.0, vgr * vgr * vgr);
                    g0 = mefec * d32 * gnode.mTot;
                }

                fx[i] -= g0 * r1;
                fy[i] -= g0 * r2;
                fz[i] -= g0 * r3;
                ugrav[i] += g0 * dd2;

#ifndef NDEBUG
                if (std::isnan(fx[i])) printf("NE i=%d fx[i]=%.15f g0=%f\n", i, fx[i], g0);
#endif
            }
            else // node is not leaf
            {

                g0 = gnode.mTot * d32; // Base Value
                fx[i] -= g0 * r1;
                fy[i] -= g0 * r2;
                fz[i] -= g0 * r3;
                ugrav[i] += g0 * dd2; // eof Base value

                const T r5 = dd2 * dd2 * dd5;
                const T r7 = r5 * dd2;

                const T qr1 = r1 * gnode.qxx + r2 * gnode.qxy + r3 * gnode.qxz;
                const T qr2 = r1 * gnode.qxy + r2 * gnode.qyy + r3 * gnode.qyz;
                const T qr3 = r1 * gnode.qxz + r2 * gnode.qyz + r3 * gnode.qzz;

                const T rqr = r1 * qr1 + r2 * qr2 + r3 * qr3;

                const T c1 = (-7.5 / r7) * rqr;
                const T c2 = 3.0 / r5;
                const T c3 = 0.5 * gnode.trq;

                fx[i] += c1 * r1 + c2 * (qr1 + c3 * r1);
                fy[i] += c1 * r2 + c2 * (qr2 + c3 * r2);
                fz[i] += c1 * r3 + c2 * (qr3 + c3 * r3);
                ugrav[i] -= (1.5 / r5) * rqr + c3 * d32;

#ifndef NDEBUG
                if (std::isnan(c1))
                {
                    printf("r7=%e dd2=%e dd5=%e r1=%e r2=%e r3=%e\n", r7, dd2, dd5, r1, r2, r3);
                    exit(0);
                }

                if (std::isnan(fx[i]))
                {
                    printf("NI, NL i=%d fx[i]=%f c1=%f c2=%f c3=%f gnode.trq=%f\n", i, fx[i], c1, c2, c3, gnode.trq);
                    exit(0);
                }
#endif
            }
        }
        else // go deeper
        {
            for (const auto &child : gnode.cells) // go deeper to the childs
            {
                if (child->global && child->assignee != child->comm_rank && child->assignee != -1) { continue; }

                treeWalkForRemoteParticlesRef(*child, i, xi, yi, zi, hi, hj, mj, fx, fy, fz, ugrav);
            }
        }
    }
}

template <typename T, typename Dataset_i, typename Dataset_j>
void gravityTreeWalkForRemoteParticlesImpl(const Task &t, const GravityOctree<T> &tree, const Dataset_j &dj, Dataset_i &di)
{
    const size_t n = t.clist.size();
    const int *clist = t.clist.data();
    const T *xi = di.x.data();
    const T *yi = di.y.data();
    const T *zi = di.z.data();
    const T *hi = di.h.data();

    const T *hj = dj.h.data();
    const T *mj = dj.m.data();

    T *fx = di.fx.data();
    T *fy = di.fy.data();
    T *fz = di.fz.data();
    T *ugrav = di.ugrav.data();

#pragma omp parallel for schedule(guided)
    for (size_t pi = 0; pi < n; ++pi)
    {
        const int i = clist[pi];
        fx[i] = fy[i] = fz[i] = ugrav[i] = 0.0;

        treeWalkForRemoteParticlesRef(tree, i, xi, yi, zi, hi, hj, mj, fx, fy, fz, ugrav);
    }
}

template <typename T, class Dataset_j, class Dataset_i>
void gravityTreeWalkForRemoteParticles(const std::vector<Task> &taskList, const GravityOctree<T> &tree, const Dataset_j &dj, Dataset_i &di)
{
    for (const auto &task : taskList)
    {
        gravityTreeWalkForRemoteParticlesImpl<T>(task, tree, dj, di);
    }
}

template <typename T>
struct GravityData
{
    GravityData()
        : noParticles(0)
    {
    }

    GravityData(size_t size)
        : x(size)
        , y(size)
        , z(size)
        , h(size)
        , fx(size)
        , fy(size)
        , fz(size)
        , ugrav(size)
        , noParticles(size)
    {
    }

    void resize(const size_t size)
    {
        resizeIn(size);
        resizeOut(size);
    }
    void resizeIn(const size_t size)
    {
        x.resize(size);
        y.resize(size);
        z.resize(size);
        h.resize(size);
    }
    void resizeOut(const size_t size)
    {
        fx.resize(size);
        fy.resize(size);
        fz.resize(size);
        ugrav.resize(size);
    }

    // in
    std::vector<T> x, y, z, h;
    // out
    std::vector<T> fx, fy, fz, ugrav;

    size_t noParticles;
};

#ifdef USE_MPI
template <typename T, class Dataset>
void remoteGravityTreeWalks(const GravityOctree<T> &tree, Dataset &d, RankToParticles &particlesForRemoteGravCalculations, bool verbose)
{
    std::ofstream nullOutput("/dev/null");
    std::ostream &output = verbose ? std::cout : nullOutput;

    MasterProcessTimer gravTimer(output, d.rank);
    gravTimer.start();

    const int rankTag = 0;
    const int particlesToSendTag = 1;
    const int dataTag = 2;
    const int noArraysToSend = 4;
    const int noRequestsPerRank = noArraysToSend + 2;
    const int sendBackRankTag = 500;
    const int sendBackParticlesTag = 501;
    const int sendBackDataTag = 502;
    const int noRanks = d.nrank;

    /*
      Check if particlesForRemoteGravCalculations map has every rank in it.
      If not, add dummy empty value. It's used later to indicate whether given rank received all requests.
      Without that rank doesn't know if it can expects more requests or can finish calculations.
     */
    if ((int)particlesForRemoteGravCalculations.size() != d.nrank - 1)
    {
        for (int i = 0; i < d.nrank; ++i)
        {
            if (i == d.rank) continue;
            const auto it = particlesForRemoteGravCalculations.find(i);
            if (it == particlesForRemoteGravCalculations.end()) particlesForRemoteGravCalculations[i].clear();
        }
    }
    // printf("[%d] particlesForRemoteGravCalculations after refinement has %lu elements\n", d.rank,
    // particlesForRemoteGravCalculations.size());

    int x = 0;
    std::vector<GravityData<T>> alldataToSend(noRanks - 1);
    std::vector<MPI_Request> requests(noRequestsPerRank * (d.nrank - 1));

    for (auto &el : particlesForRemoteGravCalculations)
    {
        const int dest = el.first;
        const size_t particlesToSendCount = static_cast<int>(el.second.size());
        const int reqStart = x * noRequestsPerRank;

        size_t toSendIdx = 0;
        auto &dataToSend = alldataToSend[x];
        dataToSend.resizeIn(particlesToSendCount);

        for (const auto i : el.second)
        {

            dataToSend.x[toSendIdx] = d.x[i];
            dataToSend.y[toSendIdx] = d.y[i];
            dataToSend.z[toSendIdx] = d.z[i];
            dataToSend.h[toSendIdx] = d.h[i];
            ++toSendIdx;
        }

        // printf("[%d] Sending %d particles\n", d.rank, particlesToSendCount);
        MPI_Isend(&d.rank, 1, MPI_INT, dest, rankTag, MPI_COMM_WORLD, &requests[reqStart + 0]);
        MPI_Isend(&particlesToSendCount, 1, MPI_INT, dest, particlesToSendTag, MPI_COMM_WORLD, &requests[reqStart + 1]);
        MPI_Isend(dataToSend.x.data(), dataToSend.x.size(), MPI_DOUBLE, dest, dataTag + 0, MPI_COMM_WORLD, &requests[reqStart + 2]);
        MPI_Isend(dataToSend.y.data(), dataToSend.y.size(), MPI_DOUBLE, dest, dataTag + 1, MPI_COMM_WORLD, &requests[reqStart + 3]);
        MPI_Isend(dataToSend.z.data(), dataToSend.z.size(), MPI_DOUBLE, dest, dataTag + 2, MPI_COMM_WORLD, &requests[reqStart + 4]);
        MPI_Isend(dataToSend.h.data(), dataToSend.h.size(), MPI_DOUBLE, dest, dataTag + 3, MPI_COMM_WORLD, &requests[reqStart + 5]);

        ++x;
    }
    gravTimer.step("    # Gravity (Remote): Isend data");

    std::map<int, int> rankToParticlesReceived;
    std::vector<MPI_Request> requests2(noRequestsPerRank * (d.nrank - 1));

    std::vector<GravityData<T>> alldataToReceive(noRanks);
    std::vector<GravityData<T>> calculatedDataToSendBack(noRanks);
    //#pragma omp parallel
    //#pragma omp single
    for (int i = 0; i < noRanks - 1; ++i)
    {
        MasterProcessTimer recvTimer(output, d.rank);
        recvTimer.start();

        MPI_Status status[noRequestsPerRank];

        int rankSource, recvParticlesCount;
        MPI_Recv(&rankSource, 1, MPI_INT, MPI_ANY_SOURCE, rankTag, MPI_COMM_WORLD, &status[0]);
        MPI_Recv(&recvParticlesCount, 1, MPI_INT, rankSource, particlesToSendTag, MPI_COMM_WORLD, &status[1]);
        // printf("[%d] Received rank %d and count %d\n", d.rank, rankSource, recvParticlesCount);

        auto &dataToReceive = alldataToReceive[rankSource];
        auto &dataToSendBack = calculatedDataToSendBack[rankSource];
        dataToReceive.resizeIn(recvParticlesCount);
        dataToSendBack.resize(recvParticlesCount);
        rankToParticlesReceived[rankSource] = recvParticlesCount;

        MPI_Recv(dataToReceive.x.data(), recvParticlesCount, MPI_DOUBLE, rankSource, dataTag + 0, MPI_COMM_WORLD, &status[2 + 0]);
        MPI_Recv(dataToReceive.y.data(), recvParticlesCount, MPI_DOUBLE, rankSource, dataTag + 1, MPI_COMM_WORLD, &status[2 + 1]);
        MPI_Recv(dataToReceive.z.data(), recvParticlesCount, MPI_DOUBLE, rankSource, dataTag + 2, MPI_COMM_WORLD, &status[2 + 2]);
        MPI_Recv(dataToReceive.h.data(), recvParticlesCount, MPI_DOUBLE, rankSource, dataTag + 3, MPI_COMM_WORLD, &status[2 + 3]);

        recvTimer.step("    # Gravity (Remote): Recv: #" + std::to_string(i));

        //#pragma omp task
        {
            MasterProcessTimer treewalkTimer(output, d.rank);
            treewalkTimer.start();

            treewalkTimer.step("    # Gravity (Remote): ForeginTreeWalk: #" + std::to_string(i) + " (" +
                               std::to_string(recvParticlesCount) + " particles)");

            dataToSendBack.x = dataToReceive.x;
            dataToSendBack.y = dataToReceive.y;
            dataToSendBack.z = dataToReceive.z;
            dataToSendBack.h = dataToReceive.h;

            auto &fd = dataToSendBack;

            std::vector<int> clist(recvParticlesCount);
            std::iota(clist.begin(), clist.end(), 0);
            const auto tl = TaskList(clist, 1, 0, 0);

            sph::gravityTreeWalkForRemoteParticles<T, Dataset, GravityData<T>>(tl.tasks, tree, d, fd);
        }
    }

    if (requests.size() > 0)
    {
        MPI_Status status[requests.size()];
        MPI_Waitall(requests.size(), &requests[0], status);
    }

    gravTimer.step("    # Gravity (Remote): calculated all foregin data");
    int k = 0;
    for (const auto &el : rankToParticlesReceived)
    {
        const auto dest = el.first;
        const auto noParticlesToSendBack = el.second;
        const auto &dataToSendBack = calculatedDataToSendBack[dest];
        const int reqStart = k++ * noRequestsPerRank;

        MPI_Isend(&d.rank, 1, MPI_INT, dest, sendBackRankTag, MPI_COMM_WORLD, &requests2[reqStart + 0]);
        MPI_Isend(&noParticlesToSendBack, 1, MPI_INT, dest, sendBackParticlesTag, MPI_COMM_WORLD, &requests2[reqStart + 1]);
        MPI_Isend(dataToSendBack.fx.data(), dataToSendBack.fx.size(), MPI_DOUBLE, dest, sendBackDataTag + 0, MPI_COMM_WORLD,
                  &requests2[reqStart + 2]);
        MPI_Isend(dataToSendBack.fy.data(), dataToSendBack.fy.size(), MPI_DOUBLE, dest, sendBackDataTag + 1, MPI_COMM_WORLD,
                  &requests2[reqStart + 3]);
        MPI_Isend(dataToSendBack.fz.data(), dataToSendBack.fz.size(), MPI_DOUBLE, dest, sendBackDataTag + 2, MPI_COMM_WORLD,
                  &requests2[reqStart + 4]);
        MPI_Isend(dataToSendBack.ugrav.data(), dataToSendBack.ugrav.size(), MPI_DOUBLE, dest, sendBackDataTag + 3, MPI_COMM_WORLD,
                  &requests2[reqStart + 5]);
    }

    std::vector<GravityData<T>> calculatedDataToReceive(noRanks);

    for (int i = 0; i < noRanks - 1; ++i)
    {
        MasterProcessTimer recvBackTimer(output, d.rank);

        MPI_Status status3[6];

        int rankSource, recvParticlesCount;

        MPI_Recv(&rankSource, 1, MPI_INT, MPI_ANY_SOURCE, sendBackRankTag, MPI_COMM_WORLD, &status3[0]);
        MPI_Recv(&recvParticlesCount, 1, MPI_INT, rankSource, sendBackParticlesTag, MPI_COMM_WORLD, &status3[1]);

        const int particlesToSendCount = recvParticlesCount;
        // printf("[%d] Received back gravity data from rank %d and count %d\n", d.rank, rankSource, particlesToSendCount);
        // fflush(stdout);
        calculatedDataToReceive[rankSource].resizeOut(particlesToSendCount);

        auto &recvFx = calculatedDataToReceive[rankSource].fx;
        auto &recvFy = calculatedDataToReceive[rankSource].fy;
        auto &recvFz = calculatedDataToReceive[rankSource].fz;
        auto &recvUgrav = calculatedDataToReceive[rankSource].ugrav;

        std::vector<T> accDataToRecv(recvParticlesCount * noArraysToSend);
        // printf("[%d] Received back gravity data from rank %d and count %d\n", d.rank, dest, recvParticlesCount);

        if (particlesToSendCount != recvParticlesCount)
            printf("[%d] ERROR: toSend =%d, recv=%d\n", d.rank, particlesToSendCount, recvParticlesCount);

        if (recvParticlesCount != 0)
        {
            MPI_Recv(recvFx.data(), particlesToSendCount, MPI_DOUBLE, rankSource, sendBackDataTag + 0, MPI_COMM_WORLD, &status3[2]);
            MPI_Recv(recvFy.data(), particlesToSendCount, MPI_DOUBLE, rankSource, sendBackDataTag + 1, MPI_COMM_WORLD, &status3[3]);
            MPI_Recv(recvFz.data(), particlesToSendCount, MPI_DOUBLE, rankSource, sendBackDataTag + 2, MPI_COMM_WORLD, &status3[4]);
            MPI_Recv(recvUgrav.data(), particlesToSendCount, MPI_DOUBLE, rankSource, sendBackDataTag + 3, MPI_COMM_WORLD, &status3[5]);
        }
        recvBackTimer.step("    # Gravity (Remote): RecvBack #" + std::to_string(i) + " (" + std::to_string(recvParticlesCount) +
                           " particles)");

        int xx = 0;
        for (const auto j : particlesForRemoteGravCalculations[rankSource])
        {

            d.fx[j] += recvFx[xx];
            d.fy[j] += recvFy[xx];
            d.fz[j] += recvFz[xx];
            d.ugrav[j] += recvUgrav[xx];
            ++xx;
        }
        recvBackTimer.step("    # Gravity (Remote): RecvBack FillData#" + std::to_string(i) + " (" + std::to_string(recvParticlesCount) +
                           " particles)");
    }

    if (requests2.size() > 0)
    {
        MPI_Status status[requests2.size()];
        MPI_Waitall(requests2.size(), &requests2[0], status);
    }
    gravTimer.step("    # Gravity (Remote): Recv back and filled calculated data");
}
#else

template <typename T, class Dataset>
void remoteGravityTreeWalks(const GravityOctree<T> &, Dataset &, RankToParticles &, bool)
{
}

#endif

} // namespace sph
} // namespace sphexa
