main() {
    try{
    let correct = ArrayList<Bool>()
            correct.append((findClosestElements(ArrayList<Float64>([1.0, 2.0, 3.9, 4.0, 5.0, 2.2])) == ArrayList<Float64>(3.9, 4.0)))
            correct.append((findClosestElements(ArrayList<Float64>([1.0, 2.0, 5.9, 4.0, 5.0])) == ArrayList<Float64>(5.0, 5.9)))
            correct.append((findClosestElements(ArrayList<Float64>([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])) == ArrayList<Float64>(2.0, 2.2)))
            correct.append((findClosestElements(ArrayList<Float64>([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])) == ArrayList<Float64>(2.0, 2.0)))
            correct.append((findClosestElements(ArrayList<Float64>([1.1, 2.2, 3.1, 4.1, 5.1])) == ArrayList<Float64>(2.2, 3.1)))
    
if (correct.contains(false)) {
                print(correct)
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}