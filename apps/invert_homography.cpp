/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#include "tpofinder/util.h"

#include <boost/program_options.hpp>

using namespace cv;
using namespace tpofinder;
using namespace std;
namespace bpo = boost::program_options;

int main(int argc, char** argv) {
    bpo::options_description options;
    string ifile, ofile;
    options.add_options()
            ("in,i", bpo::value<string > (&ifile), "file where the "
            "homography is read from.")
            ("out,o", bpo::value<string > (&ofile), "file where the inverted "
            "homography is written to.");
    bpo::positional_options_description posopts;
    posopts.add("in", 1);
    posopts.add("out", 1);

    bpo::variables_map vm;
    bpo::store(bpo::command_line_parser(argc, argv).
            options(options).positional(posopts).run(), vm);
    bpo::notify(vm);
    
    if (ifile.empty() || ofile.empty()) {
        cerr << "Usage: invert_homography <in-file> <out-file>" << endl;
        options.print(cerr);
        return -1;
    }

    invertHomography(ifile, ofile);
    
    return 0;
}

