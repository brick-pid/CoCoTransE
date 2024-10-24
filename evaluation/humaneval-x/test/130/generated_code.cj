main() {
    try{
    let correct = ArrayList<Bool>(
            tri(3) == ArrayList<Int64>(1, 3, 2, 8),
            tri(4) == ArrayList<Int64>(1, 3, 2, 8, 3),
            tri(5) == ArrayList<Int64>(1, 3, 2, 8, 3, 15),
            tri(6) == ArrayList<Int64>(1, 3, 2, 8, 3, 15, 4),
            tri(7) == ArrayList<Int64>(1, 3, 2, 8, 3, 15, 4, 24),
            tri(8) == ArrayList<Int64>(1, 3, 2, 8, 3, 15, 4, 24, 5),
            tri(9) == ArrayList<Int64>(1, 3, 2, 8, 3, 15, 4, 24, 5, 35),
            tri(20) == ArrayList<Int64>(1, 3, 2, 8, 3, 15, 4, 24, 5, 35, 6, 48, 7, 63, 8, 80, 9, 99, 10, 120, 11),
            tri(0) == ArrayList<Int64>([1]),
            tri(1) == ArrayList<Int64>(1, 3)
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