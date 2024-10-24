main() {
    try{
    let correct = ArrayList<Bool>()

            correct.append(median(ArrayList<Int64>(3, 1, 2, 4, 5)) == 3.0)
            correct.append(median(ArrayList<Int64>(-10, 4, 6, 1000, 10, 20)) == 8.0)
            correct.append(median(ArrayList<Int64>([5])) == 5.0)
            correct.append(median(ArrayList<Int64>(6, 5)) == 5.5)
if (correct.contains(false)) {
                print(correct)
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}