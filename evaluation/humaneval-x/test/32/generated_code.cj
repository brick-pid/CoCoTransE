//from std import random.*
//from std import math.*
func poly(xs: ArrayList<Float64>, x: Float64): Float64 {
    var result: Float64 = 0.0
    var i = 0

    while (i < xs.size) {
        result += xs[i] * pow(x, Float64(i))
        i += 1
    }

    result
}
main() {
    try{
        let rand = Random(42);
        for (i in 0..100) {
            var ncoeff : Int64 = 2 * (rand.nextInt64(3) + 1);
            let coeffs = ArrayList<Float64>();
            for (j in 0..ncoeff) {
                var coeff : Int64 = rand.nextInt64(20) - 10;
                if (coeff == 0) {
                    coeff = 1;
                }
                coeffs.append(Float64((coeff)));
            }
            var solution : Float64 = findZero(coeffs);
            if (abs(poly(coeffs, solution)) > 1e-4) {
                print("false")
            }
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}