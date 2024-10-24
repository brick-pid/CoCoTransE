main() {
    try{
    let correct = ArrayList<Bool>(
            evenOddPalindrome(123)==ArrayList<Int64>(8, 13),
            evenOddPalindrome(12)==ArrayList<Int64>(4, 6),
            evenOddPalindrome(3)==ArrayList<Int64>(1, 2),
            evenOddPalindrome(63)==ArrayList<Int64>(6, 8),
            evenOddPalindrome(25)==ArrayList<Int64>(5, 6),
            evenOddPalindrome(19)==ArrayList<Int64>(4, 6),
            evenOddPalindrome(9)==ArrayList<Int64>(4, 5),
            evenOddPalindrome(1)==ArrayList<Int64>(0, 1)
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