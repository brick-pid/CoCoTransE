//from std import math.*
main() {
    try{
    let correct = ArrayList<Bool>(
            abs(meanAbsoluteDeviation(ArrayList<Float64>(1.0, 2.0, 3.0)) - 2.0/3.0) < 1e-6,
            abs(meanAbsoluteDeviation(ArrayList<Float64>(1.0, 2.0, 3.0, 4.0)) - 1.0) < 1e-6,
            abs(meanAbsoluteDeviation(ArrayList<Float64>(1.0, 2.0, 3.0, 4.0, 5.0)) - 6.0/5.0) < 1e-6
    );
if (correct.contains(false)) {
                print(correct)
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}