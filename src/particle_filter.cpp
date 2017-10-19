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

#define NUM_PARTICLES 100
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

void ParticleFilter::prediction(double dt, double std_pose[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine generator;



	// predict new location for each particle
	for (int i = 0; i < num_particles; ++i) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		// avoid dividing by zero
		//if (fabs(yaw_r) > YAW_RATE_THRESHOLD) {
		if (yaw_rate != 0) {
			x += (velocity/yaw_rate)*(sin(theta+yaw_rate*dt) - sin(theta));
			y += (velocity/yaw_rate)*(cos(theta)- cos(theta+yaw_rate*dt));
			theta += yaw_rate*dt;

		}
		else {
			x += velocity*cos(theta)*dt;
			y += velocity*sin(theta)*dt;
		}
		// generate normal distribution for x
		std::normal_distribution<double> x_distribution(x,std_pose[0]);
		// generate normal distribution for y
		std::normal_distribution<double> y_distribution(y,std_pose[1]);
		// generate normal distribution for theta
		std::normal_distribution<double> theta_distribution(theta,std_pose[2]);

		particles[i].x = x_distribution(generator);
		particles[i].y = y_distribution(generator);
		particles[i].theta = theta_distribution(generator);
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
		//cout << "Sense data number" << obs_i << ":  x: " << x_obs << "  y: " << y_obs << endl;

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
				associated_landmark_ind = map_landmarks.landmark_list[lm_i].id_i;
			}
		}
		//cout << "\t map data chosen  x: " << map_landmarks.landmark_list[associated_landmark_ind].x_f << "  y: " << map_landmarks.landmark_list[associated_landmark_ind].y_f << endl;

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
		double yp = particle.y;
		double theta = particle.theta;

		// Transform the observation from particle frame to map frame
		double xm = x*cos(theta) - y*sin(theta) + xp;
		double ym = x*sin(theta) + y*cos(theta) + yp;

		// append the observations in map frame to the particle
		particle.sense_x.push_back(xm);
		particle.sense_y.push_back(ym);
	}
}

void ParticleFilter::calcWeight(const Map &map_landmarks, Particle &particle, const double std_landmark[]) {
	// compute the square of std_landmark
	const double std_landmark_2[] = {pow(std_landmark[0],2.0), pow(std_landmark[1],2.0)};
	const double const_norm = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);

	// iterate over all observations
	for (int i = 0; i < particle.associations.size(); ++i) {
		int landmark_ind = particle.associations[i] - 1;
		double x_map = map_landmarks.landmark_list[landmark_ind].x_f;
		double y_map = map_landmarks.landmark_list[landmark_ind].y_f;
		double x_obs = particle.sense_x[i];
		double y_obs = particle.sense_y[i];

		// calculate multivariate gaussian
		double exp_term = pow(x_obs-x_map,2.0)/(2*std_landmark_2[0]) + pow(y_obs-y_map,2.0)/(2*std_landmark_2[1]);
		double weight = const_norm * exp(-exp_term);
		//cout << "x_diif and y_diff are: " << fabs(x_obs-x_map) << "\t" << fabs(y_obs-y_map) << endl;
		//cout << "obs number " << i << " weight is: " << weight << endl;
		// update total weight
		particle.weight *= weight;
		//cout << "PARTICLE weight is: " << particle.weight << endl;
	}
	weights.push_back(particle.weight);
	//cout << "particle weight is: " << particle.weight << endl;
}

void ParticleFilter::updateWeights(const double sensor_range, const double std_landmark[],
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
		calcWeight(map_landmarks,particles[i],std_landmark);
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// generate discrete distribution based on the particle weights
	std::default_random_engine generator;
	std::discrete_distribution<int> w_distribution (weights.begin(),weights.end());

	std::vector<Particle> temp_particles;

	for (int i = 0; i < num_particles; ++i) {
		Particle new_particle;
		int particle_index = w_distribution(generator);
		new_particle.id = i;
		new_particle.weight = 1.0;
		new_particle.x = particles[particle_index].x;
		new_particle.y = particles[particle_index].y;
		new_particle.theta = particles[particle_index].theta;
		temp_particles.push_back(new_particle);
	}
	particles.clear();
	weights.clear();
	particles = temp_particles;
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

