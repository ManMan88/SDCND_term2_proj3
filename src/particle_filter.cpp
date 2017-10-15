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
#include <random>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define NUM_PARTICLES 1000
#define YAW_RATE_THRESHOLD 0.001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = NUM_PARTICLES;
	std::default_random_engine generator;

	// generate normal distribution for x
	std::normal_distribution<double> x_distribution(x,std[0]);

	// generate normal distribution for y
	std::normal_distribution<double> y_distribution(y,std[1]);

	// generate normal distribution for theta
	std::normal_distribution<double> theta_distribution(theta,std[2]);

	// generate particles using a normal distribution and the GPS measurement
	for (int i = 0; i < num_particles; ++i) {
		Particle init_particle;
		init_particle.id = i;
		init_particle.weight = 1.0;
		init_particle.x = x_distribution(generator);
		init_particle.y = y_distribution(generator);
		init_particle.theta = theta_distribution(generator);
		particles.push_back(init_particle);
	}

	// now the particle filter is initialized
	is_initialized = true;
	return;
}

void ParticleFilter::prediction(double dt, double std_odometry[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine generator;

	// generate normal distribution for velocity
	std::normal_distribution<double> v_distribution(velocity,std_odometry[0]);

	// generate normal distribution for yaw_rate
	std::normal_distribution<double> yaw_rate_distribution(yaw_rate,std_odometry[1]);

	// predict new location for each particle
	for (int i = 0; i < num_particles; ++i) {
		double v = v_distribution(generator);
		double yaw_r = yaw_rate_distribution(generator);
		double theta = particles[i].theta;

		// avoid dividing by zero
		if (yaw_r > YAW_RATE_THRESHOLD) {
			particles[i].x += (v/yaw_r)*(sin(theta+yaw_r*dt) - sin(theta));
			particles[i].y += (v/yaw_r)*(cos(theta)- cos(theta+yaw_r*dt));
			particles[i].theta += yaw_r*dt;
			fixAngle(particles[i].theta);
		}
		else {
			particles[i].x += v*cos(theta)*dt;
			particles[i].y += v*sin(theta)*dt;
		}
	}
}

void ParticleFilter::dataAssociation(const Map &map_landmarks, Particle &particle) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// iterate over all observations
	for (int obs_i = 0; obs_i < particle.sense_x.size(); ++obs_i) {
		// retrieve relevant variables
		double x_obs = particle.sense_x[obs_i];
		double y_obs = particle.sense_y[obs_i];

		double min_distance = 1000000000.0;
		double associated_landmark_ind = 0;

		// iterate over all landmarks
		for (int lm_i = 0; lm_i < map_landmarks.landmark_list.size(); ++lm_i) {
			// retrieve relevant variables
			double x_map = map_landmarks.landmark_list[lm_i].x_f;
			double y_map = map_landmarks.landmark_list[lm_i].y_f;

			// compute the distance between landmark and observation
			double distance = dist(x_map,y_map,x_obs,y_obs);

			// if distance is smaller, update the landmark
			if ( distance < min_distance) {
				min_distance = distance;
				associated_landmark_ind = lm_i;
			}
		}

		// append associated landmark
		particle.associations.push_back(associated_landmark_ind);
	}

}

void ParticleFilter::transformObservations(const std::vector<LandmarkObs> &observations, Particle &particle) {
	// iterate over all observations
	for (int i = 0; i < observations.size(); ++i) {
		// retrieve relevant variables
		double x = observations[i].x;
		double y = observations[i].y;
		double xp = particle.x;
		double yp = particle.x;
		double theta = particle.theta;

		// Transform the observation from particle frame to map frame
		double xm = x*cos(theta) - y*sin(theta) + xp;
		double ym = x*sin(theta) + y*sin(theta) + yp;

		// append the observations in map frame to the particle
		particle.sense_x.push_back(xm);
		particle.sense_y.push_back(ym);
	}
}

void ParticleFilter::calcWeight(const Map &map_landmarks, Particle &particle) {

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// iterate over all particles
	for (int i = 0; i < num_particles; ++i) {
		// transform all observations
		transformObservations(observations,particles[i]);

		// associate landmarks to observations
		dataAssociation(map_landmarks,particles[i]);

		// calculate the particle weight
		calcWeight(map_landmarks,particles[i]);
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

