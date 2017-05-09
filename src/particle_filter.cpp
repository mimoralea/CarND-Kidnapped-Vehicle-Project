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

        // Consult particle_filter.h for more information about this method (and others in this file).
        is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std[], double velocity, double yaw_rate) {
        // Add measurements to each particle and add random Gaussian noise.
        // When adding noise you may find std::normal_distribution and std::default_random_engine useful.
        // http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
        // http://www.cplusplus.com/reference/random/default_random_engine/

        // calculate new values
        for (int i=0; i < num_particles; i++) {
                Particle p = particles[i];
                double t_new = p.theta + yaw_rate * delta_t;
                if (fabs(yaw_rate) > 0.001) {
                        p.x += velocity / yaw_rate * (sin(p.theta + t_new) - sin(p.theta));
                        p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + t_new));
                } else {
                        p.x += velocity * delta_t * cos(p.theta);
                        p.y += velocity * delta_t * sin(p.theta);
                }
                p.theta = t_new;

                // add random Gaussian noise
                addGaussianNoise(p, std);
                particles[i] = p;
        }
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

        // prepare predicted landmarks
        vector<LandmarkObs> predicted_observations(map_landmarks.landmark_list.size());
        for (unsigned i = 0; i < predicted_observations.size(); i++) {
                predicted_observations[i].id = map_landmarks.landmark_list[i].id_i;
                predicted_observations[i].x = map_landmarks.landmark_list[i].x_f;
                predicted_observations[i].y = map_landmarks.landmark_list[i].y_f;
        }

        // iterate over each particle observations
        vector<LandmarkObs> particle_observations(observations.size());
        for (unsigned i = 0; i < num_particles; i++) {
                Particle p = particles[i];

                // transform observations from vehicles coordinate system to map
                for (unsigned j = 0; j < observations.size(); j++) {
                        LandmarkObs o = observations[j];
                        particle_observations[j].id = o.id;
                        particle_observations[j].x = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
                        particle_observations[j].y = p.y + o.y * cos(p.theta) + o.x * sin(p.theta);
                }

                // associate landmarks and observations
                dataAssociation(predicted_observations, particle_observations);

                // recalculate weights
                p.weight = 1.;
                for (unsigned j = 0; j < particle_observations.size(); j++) {
                        LandmarkObs obs = particle_observations[j];
                        LandmarkObs pred = predicted_observations[obs.id];

                        double dx = obs.x - pred.x;
                        double dy = obs.y - pred.y;
                        double std_x = std_landmark[0];
                        double std_y = std_landmark[1];

                        p.weight *= exp(-dx * dx / (2 * std_x * std_x) - dy * dy / (2 * std_y * std_y))/ (2 * M_PI * std_x * std_y);
                }
                particles[i] = p;
                weights[i] = p.weight;
        }
}

void ParticleFilter::resample() {
        // Resample particles with replacement with probability proportional to their weight.
        // You may find std::discrete_distribution helpful here.
        // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        vector<Particle> new_particles(particles.size());
        for (int i = 0; i < num_particles; i++) {
                new_particles[i] = particles[dist(generator)];
        }
        particles = new_particles;
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
