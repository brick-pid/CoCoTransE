main() {
    try{
    let correct = ArrayList<Bool>(
            solution(ArrayList<Int64>(5, 8, 7, 1)) == 12,
            solution(ArrayList<Int64>(3, 3, 3, 3, 3)) == 9,
            solution(ArrayList<Int64>(30, 13, 24, 321)) == 0,
            solution(ArrayList<Int64>(5, 9)) == 5,
            solution(ArrayList<Int64>(2, 4, 8)) == 0,
            solution(ArrayList<Int64>(30, 13, 23, 32)) == 23,
            solution(ArrayList<Int64>(3, 13, 2, 9)) == 3
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