//! Outils pour les threads de calcul : verrous qui survivent a une panique, et jonction qui la propage.

use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::ScopedJoinHandle;

/// Prend le verrou, meme si un thread a panique en le tenant.
///
/// Les donnees protegees ici (files de taches, caches) restent coherentes entre deux operations :
/// un verrou « empoisonne » ne doit pas faire tomber a son tour chaque thread qui le prend.
pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Attend sur `condvar` en relachant `guard`, avec la meme tolerance que [`lock`].
pub fn wait<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    condvar.wait(guard).unwrap_or_else(PoisonError::into_inner)
}

/// Resultat d'un thread d'un `std::thread::scope`. Si le thread a panique, la panique reprend dans
/// l'appelant avec son message d'origine, au lieu d'etre remplacee par celui d'un `unwrap`.
pub fn join<T>(handle: ScopedJoinHandle<'_, T>) -> T {
    handle.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_survives_a_panic_in_another_thread() {
        let mutex = Mutex::new(1);
        let panicked = std::thread::scope(|s| {
            s.spawn(|| {
                let _guard = mutex.lock().unwrap();
                panic!("panique voulue par le test");
            })
            .join()
        });
        assert!(panicked.is_err());
        assert!(mutex.is_poisoned());
        *lock(&mutex) += 1;
        assert_eq!(*lock(&mutex), 2);
    }

    #[test]
    fn join_returns_the_value_of_the_thread() {
        let v = std::thread::scope(|s| join(s.spawn(|| 6 * 7)));
        assert_eq!(v, 42);
    }
}
