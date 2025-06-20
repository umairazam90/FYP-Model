<?php
require_once __DIR__ . '/includes/db_config.php';
require_once __DIR__ . '/includes/header.php'; 

$message = '';
$message_type = '';

if (isset($_GET['registration']) && $_GET['registration'] == 'success') {
    $message = "Registration successful! Please log in.";
    $message_type = "success";
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username_email = trim($_POST['username_email']);
    $password = $_POST['password'];

    $stmt = $conn->prepare("SELECT user_id, username, password, role_id FROM users WHERE username = ? OR email = ?");
    $stmt->bind_param("ss", $username_email, $username_email);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows == 1) {
        $user = $result->fetch_assoc();
        if (password_verify($password, $user['password'])) {
            $_SESSION['user_id'] = $user['user_id'];
            $_SESSION['username'] = $user['username'];
            $_SESSION['role_id'] = $user['role_id'];

            if ($user['role_id'] == 1) { 
                header("Location: " . BASE_URL . "index.php");
            } elseif ($user['role_id'] == 2) { 
                header("Location: " . BASE_URL . "admin/index.php");
            }
            exit();
        } else {
            $message = "Invalid password.";
            $message_type = "error";
        }
    } else {
        $message = "User not found. Please check your username/email.";
        $message_type = "error";
    }
    $stmt->close();
}
$conn->close();
?>

<div class="card shadow-sm p-4">
    <h2 class="card-title text-center mb-4 text-primary"><i class="fas fa-sign-in-alt me-2"></i>Login to Your Account</h2>

    <?php if ($message): ?>
        <div class="alert <?php echo ($message_type == 'success') ? 'alert-success' : 'alert-danger'; ?>" role="alert">
            <?php echo htmlspecialchars($message); ?>
        </div>
    <?php endif; ?>

    <form action="login.php" method="POST">
        <div class="mb-3">
            <label for="username_email" class="form-label">Username or Email:</label>
            <input type="text" class="form-control" id="username_email" name="username_email" required>
        </div>
        <div class="mb-4">
            <label for="password" class="form-label">Password:</label>
            <input type="password" class="form-control" id="password" name="password" required>
        </div>
        <div class="d-grid gap-2">
            <button type="submit" class="btn btn-primary btn-lg">Login</button>
        </div>
    </form>

    <p class="text-center mt-3">Don't have an account? <a href="<?php echo BASE_URL; ?>register.php" class="text-decoration-none">Register here</a>.</p>
</div>

<?php
require_once __DIR__ . '/includes/footer.php';
?>
