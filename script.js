document.addEventListener('mousemove', (e) => {
    const cube = document.querySelector('.cube');
    const x = (e.clientX / window.innerWidth) * 360;
    const y = (e.clientY / window.innerHeight) * 360;
    cube.style.transform = `rotateY(${x}deg) rotateX(${-y}deg)`;
  });
  