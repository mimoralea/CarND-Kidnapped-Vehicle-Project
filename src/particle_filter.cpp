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

using std::normal_distribution;
using std::default_random_engine;

default_random_engine generator;

void ParticleFilter::addGaussianNoise(Particle& p, double std[]) {

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

  // Add random Gaussian noise to each particle.
  for (int i=0; i < num_particles; i++) {

    // create particle
    Particle p;
    p.id = i;

    p.x = x;
    p.y = y;
    p.theta = theta;

    std::cout << "Before:" << std::endl;
    std::cout << p.x << std::endl;
    std::cout << p.y << std::endl;
    std::cout << p.theta << std::endl;
    addGaussianNoise(p, std);

    std::cout << "After:" << std::endl;
    std::cout << p.x << std::endl;
    std::cout << p.y << std::endl;
    std::cout << p.theta << std::endl;
    std::cout << std::endl;

    p.weight = 1.0;

    // add particle to vector containing all particles
    particles.push_back(p);
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
    double td = yaw_rate * delta_t;
    if (fabs(yaw_rate) > 0.0001) {
      p.x += velocity / yaw_rate * (sin(p.theta + td) - sin(p.theta));
      p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + td));
    } else {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }
    p.theta += td;

    // add random Gaussian noise
    addGaussianNoise(p, std);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
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
