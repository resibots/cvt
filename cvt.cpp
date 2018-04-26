// C++ code for creating a CVT
// Vassilis Vassiliades - Inria, Nancy - April 2018

#include <iostream>
#include <vector>
#include <numeric>   //for iota
#include <algorithm> //for random_shuffle
#include <fstream>

#include <boost/program_options.hpp>
#include <tbb/tbb.h>
#include <Eigen/Core>

namespace cvt
{
struct SampledInitializer
{
    static Eigen::MatrixXd init(const Eigen::MatrixXd &data, const size_t num_clusters)
    {
        // number of instances = data.rows()
        assert((int)num_clusters <= data.rows());

        std::vector<size_t> indices(data.rows());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end());

        // Create the centroids
        size_t dim = data.cols();
        Eigen::MatrixXd centroids = Eigen::MatrixXd::Zero(num_clusters, dim);

        for (size_t i = 0; i < num_clusters; ++i)
            centroids.row(i) = data.row(indices[i]);

        return centroids;
    }
};

struct EuclideanDistance
{
    static double evaluate(const Eigen::VectorXd &p1, const Eigen::VectorXd &p2)
    {
        return (p1 - p2).norm();
    }
};

template <typename DistanceMetric, typename Initializer>
class KMeans
{
  public:
    KMeans(const size_t max_iterations=100, const size_t restarts=1, const double tolerance = 10e-8) : _max_iterations(max_iterations), _restarts(restarts), _tolerance(tolerance)
    {
    }

    const Eigen::MatrixXd &cluster(const Eigen::MatrixXd &data, const size_t num_clusters)
    {
        std::vector<Eigen::MatrixXd> all_centroids(_restarts, Eigen::MatrixXd::Zero(num_clusters, data.cols()));
        std::vector<double> all_losses(_restarts, 0.0);

        for (size_t r = 0; r < _restarts; ++r)
        {
            std::cout << "-----------------------------------------------------" << std::endl;
            std::cout << "  Run " << r << std::endl;
            std::cout << "-----------------------------------------------------" << std::endl;
            
            // Initialize
            all_centroids[r] = Initializer::init(data, num_clusters);

            // Iterate (EM)
            double loss, prev_loss;
            loss = prev_loss = 0.0;
            double delta = _tolerance;

            for (size_t i = 0; i < _max_iterations; ++i)
            {
                Eigen::MatrixXd new_centroids = Eigen::MatrixXd::Zero(num_clusters, data.cols());

                // Calculate the distances
                std::vector<size_t> counts(num_clusters, 0);
                loss = _calc_distances(data, all_centroids[r], new_centroids, counts);

                // delta = fabs(prev_loss - loss) / loss;
                delta = fabs(prev_loss - loss);

                if (i == 0)
                    std::cout << "Iteration " << i << " -> Loss(" << i << "): " << loss << std::endl;
                else
                    // std::cout << "Iteration " << i << " -> Loss(" << i << "): " << loss << " -> |Loss(" << i << ") - Loss(" << i - 1 << ")|: " << delta << std::endl;
                    std::cout << "Iteration " << i << " -> Loss(" << i << "): " << loss << " -> delta: " << delta << std::endl;

                if (delta < _tolerance)
                {
                    std::cout << "delta < tolerance. breaking..." << std::endl;
                    break;
                }

                prev_loss = loss;

                // Update the centroids
                _update_centroids(new_centroids, counts);

                all_centroids[r] = new_centroids;
            }

            // Store this centroid and the loss
            all_losses[r] = loss;
        }

        // Return the centroids with the lowest loss
        size_t argmin_index = std::distance(all_losses.begin(), std::min_element(all_losses.begin(), all_losses.end()));

        std::cout << "Choosing centroids from run " << argmin_index << std::endl;

        _centroids = all_centroids[argmin_index];

        return _centroids;
    }

  protected:
    double _calc_distances(const Eigen::MatrixXd &data, const Eigen::MatrixXd &centroids, Eigen::MatrixXd &new_centroids, std::vector<size_t> &counts)
    {
        size_t nb_points = data.rows();
        double sum = 0.0;
        static tbb::mutex sm;

        tbb::parallel_for(size_t(0), nb_points, size_t(1), [&](size_t i) {
            // Find the closest centroid to this point.
            double min_distance = std::numeric_limits<double>::infinity();
            size_t closest_cluster = centroids.rows(); // Invalid value.

            for (int j = 0; j < centroids.rows(); j++)
            {
                const double distance =
                    DistanceMetric::evaluate(data.row(i), centroids.row(j));

                if (distance < min_distance)
                {
                    min_distance = distance;
                    closest_cluster = j;
                }

                // Since the minimum distance cannot be less than 0
                // we could accelerate computation by breaking
                if (min_distance == 0.0)
                    break;
            }

            tbb::mutex::scoped_lock lock; // create a lock
            lock.acquire(sm);
            sum += min_distance;
            // We now have the minimum distance centroid index.
            new_centroids.row(closest_cluster) += data.row(i);
            counts[closest_cluster]++;
            lock.release();
        });

        // The loss is the mean
        return sum / (double)nb_points;
    }

    void _update_centroids(Eigen::MatrixXd &new_centroids, const std::vector<size_t> &counts)
    {
        // TODO: vectorize
        for (int i = 0; i < new_centroids.rows(); ++i)
        {
            new_centroids.row(i) = new_centroids.row(i) / (double)counts[i];
        }
    }

    size_t _max_iterations;
    size_t _restarts;
    double _eta;
    double _tolerance;
    Eigen::MatrixXd _centroids;
};
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    namespace po = boost::program_options;
    po::options_description desc("Available options");
    desc.add(boost::program_options::options_description());
    desc.add_options()("help,h", "produce help message")(
        "dimensionality,d", po::value<size_t>(), "dimensionality")(
        "centroids,c", po::value<size_t>(), "number of centroids")(
        "numsamples,n", po::value<size_t>(), "number of sampled points")(
        "iterations,i", po::value<size_t>(), "maximum number of kmeans iterations")(
        "restarts,r", po::value<size_t>(), "number of kmeans restarts")(
        "tolerance,t", po::value<double>(), "tolerance level");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc;
        return 1;
    }

    size_t number_of_sample_points = 100000;
    size_t nb_centroids = 7;
    size_t max_iterations = 100;
    size_t dimensionality = 2;
    size_t restarts = 1;
    double tolerance = 10e-8;

    if (vm.count("dimensionality"))
        dimensionality = vm["dimensionality"].as<size_t>();
    if (vm.count("centroids"))
        nb_centroids = vm["centroids"].as<size_t>();
    if (vm.count("numsamples"))
        number_of_sample_points = vm["numsamples"].as<size_t>();
    if (vm.count("iterations"))
        max_iterations = vm["iterations"].as<size_t>();
    if (vm.count("restarts"))
        restarts = vm["restarts"].as<size_t>();
    if (vm.count("tolerance"))
        tolerance = vm["tolerance"].as<double>();

    // Checks
    if (dimensionality >= number_of_sample_points)
    {
        std::cout << "The dimensionality (" << dimensionality << ") must be less than the number of samples (" << number_of_sample_points << ")." << std::endl;

        return 1;
    }

    if (nb_centroids > number_of_sample_points)
    {
        std::cout << "The number of centroids (" << nb_centroids << ") must be less than or equal to the number of samples (" << number_of_sample_points << ")." << std::endl;

        return 1;
    }

    if (max_iterations <= 0)
    {
        std::cout << "The maximum number of kmeans iterations (" << max_iterations << ") must be greater than 0." << std::endl;

        return 1;
    }

    if (restarts <= 0)
    {
        std::cout << "The number of kmeans restarts (" << restarts << ") must be greater than 0." << std::endl;

        return 1;
    }

    if (tolerance <= 0.0)
    {
        std::cout << "The tolerance (" << tolerance << ") must be greater than 0." << std::endl;

        return 1;
    }

    std::cout << "Using:";
    std::cout << "\n dimensionality = " << dimensionality;
    std::cout << "\n number of centroids = " << nb_centroids;
    std::cout << "\n number of sampled points = " << number_of_sample_points;
    std::cout << "\n maximum kmeans iterations = " << max_iterations;
    std::cout << "\n number of restarts = " << restarts;
    std::cout << "\n tolerance = " << tolerance;
    std::cout << std::endl;

    // The dataset we are clustering.
    // Random numbers in [0,1] (Eigen creates in [-1,1])
    Eigen::MatrixXd data = (Eigen::MatrixXd::Random(number_of_sample_points, dimensionality) + Eigen::MatrixXd::Constant(number_of_sample_points, dimensionality, 1.0)) / 2.0;

    cvt::KMeans<cvt::EuclideanDistance, cvt::SampledInitializer> k(max_iterations, restarts, tolerance);

    Eigen::MatrixXd centroids = k.cluster(data, nb_centroids);

    std::string filename = "centroids_" + std::to_string(nb_centroids) + "_" + std::to_string(dimensionality) + ".dat";
    std::ofstream fout(filename);

    if (fout)
    {
        std::cout << "Writing to " << filename << " ... ";
        fout << centroids << std::endl;
        std::cout << "Done!" << std::endl;
        fout.close();
    }

    return 0;
}
