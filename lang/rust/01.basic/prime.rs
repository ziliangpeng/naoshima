fn is_prime_bruteforce(n: u64) -> bool {
    if n <= 1 {
        return false;
    }
    for i in 2..n {
        if n % i == 0 {
            return false;
        }
    }
    true
}

fn all_primes(n: u64) -> Vec<u64> {
    let mut primes = Vec::new();
    for i in 2..n {
        let mut is_prime = true;
        for p in &primes {
            if i % p == 0 {
                is_prime = false;
                break;
            }
        }
        if is_prime {
            primes.push(i);
        }
    }
    primes
}

fn main() {
    for i in 2..42 {
        if is_prime_bruteforce(i) {
            print!("{} ", i);
        }
    }
    println!();
    println!("{:?}", all_primes(42));
}
