/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <map>

#include "particle_filter.h"
#include "helper_functions.h"

using std::normal_distribution;
using std::default_random_engine;
using std::numeric_limits;
using std::vector;

default_random_engine generator;


void addGaussianNoise(Particle& p, double std[]) {

        // extract standard deviations
        double x_std = std[0];
        double y_std = std[1];
        double t_std = std[2];

        // Initialize all particles to first position (based on estimates of x, y,
        // theta and their uncertainties from GPS) and all weights to 1.
        normal_distribution<double> xs(p.x,  x_std);
        normal_distribution<double> ys(p.y,  y_std);
        normal_distribution<double> ts(p.theta,  t_std);

        p.x = xs(generator);
        p.y = ys(generator);
        p.theta = ts(generator);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {

        // get out if already initialized
        if(is_initialized) {
                return;
        }

        // Set the number of particles.
        num_particles = 100;

        particles.clear();
        weights.clear();

        // Add random Gaussian noise to each particle.
        for (int i=0; i < num_particles; i++) {

                // create particle
                Particle p;
                p.id = i;

                p.x = x;
                p.y = y;
                p.theta = theta;
                addGaussianNoise(p, std);

                p.weight = 1.;

                // add particle to vector containing all particles
                particles.push_back(p);
                weights.push_back(p.weight);

        }

        /*
        // debug statements
        for (unsigned int i = 0; i < num_particles; i++) {
                Particle p = particles[i];
                double w = weights[i];
                std::cout << p.x << ", " << p.y << ", " << p.theta << ", " << p.weight << ", " << w << std::endl;
        }
        exit(1);
        */

        // Consult particle_filter.h for more information about this method (and others in this file).
        is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std[], double velocity, double yaw_rate) {
        // Add measurements to each particle and add random Gaussian noise.
        // When adding noise you may find std::normal_distribution and std::default_random_engine useful.
        // http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
        // http://www.cplusplus.com/reference/random/default_random_engine/
        /*
        std::cout << "Measurements:" << std::endl;
        std::cout << delta_t << ", " <<
                std[0] << ", " <<
                std[1] << ", " <<
                std[2] << ", " <<
                velocity << ", " <<
                yaw_rate << std::endl << std::endl;
        */

        // calculate new values
        double x, y, theta;
        for (int i=0; i < num_particles; i++) {
                Particle p = particles[i];
                //std::cout << "Original particle:" << std::endl;
                //std::cout << p.x << ", " << p.y << ", " << p.theta << ", " << p.weight << std::endl;

                theta = p.theta + yaw_rate * delta_t;
                if (fabs(yaw_rate) > 0.001) {
                        x = p.x + velocity / yaw_rate * (sin(theta) - sin(p.theta));
                        y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(theta));
                } else {
                        x = p.x + velocity * delta_t * cos(p.theta);
                        y = p.y + velocity * delta_t * sin(p.theta);
                }

                //std::cout << "Predicted no-noise particle:" << std::endl;
                //std::cout << x << ", " << y << ", " << theta << ", " << p.weight << std::endl;

                // add random Gaussian noise
                p.x = x;
                p.y = y;
                p.theta = theta;
                addGaussianNoise(p, std);
                particles[i] = p;

                //std::cout << "Predicted particle:" << std::endl;
                //std::cout << p.x << ", " << p.y << ", " << p.theta << ", " << p.weight << std::endl << std::endl;
        }
        // exit(1);
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
        // Find the predicted measurement that is closest to each observed measurement and assign the
        // observed measurement to this particular landmark.
        // this method will NOT be called by the grading code. But you will probably find it useful to
        // implement this method and use it as a helper during the updateWeights phase.
        for (unsigned int i = 0; i < observations.size(); i++) {
                double min_dist = numeric_limits<double>::infinity();
                for (unsigned int j = 0; j < predicted.size(); j++) {
                        double distance = dist(observations[i].x, observations[i].y,
                                               predicted[j].x, predicted[j].y);
                        if (min_dist > distance) {
                                observations[i].id = j;
                                min_dist = distance;
                        }
                }
        }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
        // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
        // more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        // The observations are given in the VEHICLE'S coordinate system. Your particles are located
        // according to the MAP'S coordinate system. You will need to transform between the two systems.
        // Keep in mind that this transformation requires both rotation AND translation (but no scaling).
        // The following is a good resource for the theory:
        // https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
        // and the following is a good resource for the actual equation to implement (look at equation
        // 3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
        // for the fact that the map's y-axis actually points downwards.)
        // http://planning.cs.uiuc.edu/node99.html
        /*
        std::cout << "Values passed to updateWeights:" << std::endl;
        std::cout << sensor_range << ", " <<
                std_landmark[0] << ", " <<
                std_landmark[1] << ", " <<
                std_landmark[2] << ", " <<
                observations.size() << ", " <<
                map_landmarks.landmark_list.size() << std::endl << std::endl;

        std::cout << "Observations:" << std::endl;
        for (unsigned i = 0; i < observations.size(); i++) {
                std::cout << observations[i].id << ", " <<
                        observations[i].x << ", " <<
                        observations[i].y << ", " << std::endl;
        }
        std::cout << "Total observations: " << std::endl << std::endl;
        */

        // iterate over each particle and update all weights
        for (unsigned i = 0; i < num_particles; i++) {
                Particle p = particles[i];

                // prepare landmarks only in range
                // std::cout << "Adding in-range landmarks:" << std::endl;
                vector<LandmarkObs> in_range_landmarks;
                for (unsigned j = 0; j < map_landmarks.landmark_list.size(); j++) {
                        Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
                        if (dist(p.x, p.y, landmark.x_f, landmark.y_f) > sensor_range) {
                                continue;
                        }

                        LandmarkObs obs;
                        obs.id = landmark.id_i;
                        obs.x = landmark.x_f;
                        obs.y = landmark.y_f;
                        //std::cout << obs.id << ", " << obs.x << ", " << obs.y << std::endl;
                        in_range_landmarks.push_back(obs);
                }
                //std::cout << "Total landmarks in-range: " << in_range_landmarks.size() << std::endl << std::endl;

                // transform observations from vehicles coordinate system to map
                //std::cout << "Transformed LandmarkObs:" << std::endl;
                vector<LandmarkObs> observed_landmarks(observations.size());
                for (unsigned j = 0; j < observations.size(); j++) {
                        LandmarkObs o = observations[j];
                        observed_landmarks[j].id = -1;
                        observed_landmarks[j].x = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
                        observed_landmarks[j].y = p.y + o.y * cos(p.theta) + o.x * sin(p.theta);
                        /*
                        std::cout << o.id << ", " << o.x << ", " << o.y << " ====> " <<
                                observed_landmarks[j].id << ", " <<
                                observed_landmarks[j].x << ", " <<
                                observed_landmarks[j].y << std::endl;
                        */
                }
                //std::cout << "Total observed landmarks: " << observed_landmarks.size() << std::endl << std::endl;

                // associate landmarks and observations
                dataAssociation(in_range_landmarks, observed_landmarks);

                /*
                std::cout << "Mapped LandmarkObs:" << std::endl;
                for (unsigned j = 0; j < observed_landmarks.size(); j++) {
                        std::cout << observed_landmarks[j].id << ", " <<
                                observed_landmarks[j].x << ", " <<
                                observed_landmarks[j].y << std::endl;
                }
                std::cout << "Total mapped landmarks: " << observed_landmarks.size() << std::endl << std::endl;

                std::cout << "Weights before:" << std::endl;
                for (unsigned j = 0; j < weights.size(); j++) {
                        std::cout << weights[j] << std::endl;
                }
                std::cout << "Total weights: " << weights.size() << std::endl << std::endl;
                */

                // recalculate weights
                p.weight = 1.;
                for (unsigned j = 0; j < observed_landmarks.size(); j++) {
                        LandmarkObs obs = observed_landmarks[j];
                        LandmarkObs pred = in_range_landmarks[obs.id];

                        /*
                        std::cout << obs.id << ", " <<
                                obs.x << ", " <<
                                obs.y << ", " << " -> " <<
                                pred.id << ", " <<
                                pred.x << ", " <<
                                pred.y << std::endl;
                        */

                        double dx = obs.x - pred.x;
                        double dy = obs.y - pred.y;
                        double std_x = std_landmark[0];
                        double std_y = std_landmark[1];

                        p.weight *= exp(-dx * dx / (2 * std_x * std_x) - dy * dy / (2 * std_y * std_y))/ (2 * M_PI * std_x * std_y);
                }
                // std::cout << "New particle weight ==> " << p.weight << std::endl << std::endl;
                particles[i] = p;
                weights[i] = p.weight;
        }

        double sum = 0.0;
        for (unsigned int i = 0; i < weights.size(); i++) {
                sum += weights[i];
        }
        //std::cout << "weights sum before normalizing " << sum << std::endl;
        for(unsigned int i = 0; i < particles.size(); i++) {
                double weight = particles[i].weight / sum;
                particles[i].weight = weight;
                weights[i] = weight;
        }

        sum = 0.0;
        for (unsigned int i = 0; i < particles.size(); i++) {
                sum += particles[i].weight;
        }
        //std::cout << "particles weights sum after normalization " << sum << std::endl;

        sum = 0.0;
        for (unsigned int i = 0; i < weights.size(); i++) {
                sum += weights[i];
                // std::cout << weights[i] << std::endl;
        }
        // std::cout << "weights sum after normalization " << sum << std::endl;
}

void ParticleFilter::resample() {
        // Resample particles with replacement with probability proportional to their weight.
        // You may find std::discrete_distribution helpful here.
        // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

        // std::cout << "Selecting particles according to their weights" << std::endl;
        std::map<int, int> counts;
        std::discrete_distribution<int> distribution(weights.begin(), weights.end());
        vector<Particle> new_particles(particles.size());
        for (int i = 0; i < num_particles; i++) {
                Particle chosen = particles[distribution(generator)];
                /*
                std::cout << chosen.id << ", " <<
                        chosen.x << ", " <<
                        chosen.y << ", " <<
                        chosen.theta << ", " <<
                        chosen.weight << std::endl;
                */
                counts[chosen.id] += 1;
                new_particles[i] = chosen;
        }
        particles = new_particles;
        //std::cout << std::endl;

        /*
        std::cout << "Particle distributions:" << std::endl;
        for (const auto &myPair : counts) {
                std::cout << myPair.first << ": " << myPair.second << std::endl;
        }
        std::cout << std::endl;
        */
}

void ParticleFilter::write(std::string filename) {
        // You don't need to modify this file.
        std::ofstream dataFile;
        dataFile.open(filename, std::ios::app);
        for (int i = 0; i < num_particles; ++i) {
                dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
        }
        dataFile.close();
}
