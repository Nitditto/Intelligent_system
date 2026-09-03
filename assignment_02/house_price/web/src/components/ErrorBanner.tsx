import React from "react";

interface ErrorBannerProps {
  message: string;
  onDismiss?: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({ message, onDismiss }) => {
  if (!message) return null;

  return (
    <div className="error-banner" role="alert">
      <div className="error-content">
        <div className="error-text">
          <strong>Error Notification:</strong>
          <p>{message}</p>
        </div>
      </div>
      {onDismiss && (
        <button type="button" className="error-close" onClick={onDismiss} aria-label="Close">
          Dismiss
        </button>
      )}
    </div>
  );
};
