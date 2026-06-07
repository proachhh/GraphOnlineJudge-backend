package com.myoj.webview;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.KeyEvent;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private WebView webView;
    private final Handler handler = new Handler(Looper.getMainLooper());

    // ============================================================
    // 配置区：修改下面的 URL 指向你的服务器
    // ============================================================
    private static final String BASE_URL = "http://47.108.158.48";
    // 开发环境本地测试用： "http://10.0.2.2:80"
    // 生产环境用：        "http://你的服务器IP"
    // ============================================================

    @Override
    @SuppressLint("SetJavaScriptEnabled")
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        webView = findViewById(R.id.webview);

        // WebView 设置
        webView.getSettings().setJavaScriptEnabled(true);
        webView.getSettings().setDomStorageEnabled(true);
        webView.getSettings().setDatabaseEnabled(true);
        webView.getSettings().setUseWideViewPort(true);
        webView.getSettings().setLoadWithOverviewMode(true);
        webView.getSettings().setSupportZoom(true);
        webView.getSettings().setBuiltInZoomControls(true);
        webView.getSettings().setDisplayZoomControls(false);
        webView.getSettings().setAllowFileAccess(false);
        webView.getSettings().setAllowContentAccess(false);

        // 支持混合内容（HTTP + HTTPS）
        webView.getSettings().setMixedContentMode(
                android.webkit.WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);

        // WebViewClient：在应用内加载页面，不跳转外部浏览器
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                view.loadUrl(request.getUrl().toString());
                return true;
            }

            @Override
            public void onReceivedError(WebView view, int errorCode,
                                        String description, String failingUrl) {
                showErrorPage();
            }
        });

        // WebChromeClient：支持 JavaScript 弹窗等
        webView.setWebChromeClient(new WebChromeClient());

        // 加载页面
        if (isNetworkAvailable()) {
            webView.loadUrl(BASE_URL);
        } else {
            showNetworkDialog();
        }
    }

    /**
     * 返回到上一页，如果无法返回则退出
     */
    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK && webView.canGoBack()) {
            webView.goBack();
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    @Override
    protected void onDestroy() {
        if (webView != null) {
            webView.destroy();
        }
        super.onDestroy();
    }

    /**
     * 检查网络是否可用
     */
    private boolean isNetworkAvailable() {
        ConnectivityManager cm = (ConnectivityManager) getSystemService(CONNECTIVITY_SERVICE);
        if (cm == null) return false;
        Network network = cm.getActiveNetwork();
        if (network == null) return false;
        NetworkCapabilities caps = cm.getNetworkCapabilities(network);
        return caps != null && (caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
                caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) ||
                caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET));
    }

    private void showNetworkDialog() {
        new AlertDialog.Builder(this)
                .setTitle("网络不可用")
                .setMessage("请检查网络连接后重试")
                .setPositiveButton("重试", (d, w) -> {
                    if (isNetworkAvailable()) {
                        webView.loadUrl(BASE_URL);
                    } else {
                        showNetworkDialog();
                    }
                })
                .setNegativeButton("退出", (d, w) -> finish())
                .setCancelable(false)
                .show();
    }

    private void showErrorPage() {
        handler.post(() -> webView.loadUrl("about:blank"));
    }
}
