FROM nginx:alpine

# Copy website files to nginx html directory
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/
COPY main.js /usr/share/nginx/html/
COPY tfjs_model/ /usr/share/nginx/html/tfjs_model/

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Render uses PORT env variable
EXPOSE 10000

CMD ["nginx", "-g", "daemon off;"]
