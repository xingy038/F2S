// ============================================
// Navbar Scroll Effect
// ============================================
window.addEventListener('scroll', function() {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// ============================================
// Smooth Scrolling for Navigation Links
// ============================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href !== '#' && href.length > 1) {
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const navbarHeight = document.getElementById('navbar').offsetHeight;
                const targetPosition = target.offsetTop - navbarHeight - 20;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        }
    });
});

// ============================================
// Copy BibTeX Citation
// ============================================
function copyBibTeX() {
    const citationText = `@inproceedings{synthverse2025,
  title={SynthVerse: A Large-Scale Diverse Synthetic Dataset for Point Tracking},
  author={Zhao, Weiguang and Xu, Haoran and Miao, Xingyu and Zhao, Qin and Zhang, Rui and Huang, Kaizhu and Gao, Ning and Cao, Peizhou and Sun, Mingze and Yu, Mulin and Lu, Tao and Xu, Linning and Dong, Junting and Pang, Jiangmiao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}`;

    const copyBtn = document.querySelector('.copy-btn');
    const originalHTML = copyBtn.innerHTML;

    function showSuccess() {
        copyBtn.innerHTML = `
            <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                <path d="M10.97 4.97a.75.75 0 0 1 1.07 1.05l-3.99 4.99a.75.75 0 0 1-1.08.02L4.324 8.384a.75.75 0 1 1 1.06-1.06l2.094 2.093 3.473-4.425a.267.267 0 0 1 .02-.022z"/>
            </svg>
            Copied!
        `;
        setTimeout(() => {
            copyBtn.innerHTML = originalHTML;
        }, 2000);
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(citationText).then(showSuccess).catch(() => {
            fallbackCopy(citationText, showSuccess);
        });
    } else {
        fallbackCopy(citationText, showSuccess);
    }
}

function fallbackCopy(text, onSuccess) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    try {
        document.execCommand('copy');
        onSuccess();
    } catch (err) {
        alert('Failed to copy citation. Please copy manually.');
    }
    document.body.removeChild(textarea);
}

// ============================================
// Fade-in Animation on Scroll
// ============================================
function fadeInOnScroll() {
    const elements = document.querySelectorAll('.feature-card, .stat-card, .scene-type-card, .finding-card, .download-card, .video-container, .pipeline-step, .table-wrapper');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '0';
                entry.target.style.transform = 'translateY(20px)';
                entry.target.style.transition = 'opacity 0.6s ease, transform 0.6s ease';

                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }, 100);

                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1
    });

    elements.forEach(element => {
        observer.observe(element);
    });
}

// ============================================
// Active Navigation Link Highlighting
// ============================================
function updateActiveNavLink() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';
        const navbarHeight = document.getElementById('navbar').offsetHeight;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (window.scrollY >= (sectionTop - navbarHeight - 100)) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// ============================================
// Lazy Loading Images
// ============================================
function lazyLoadImages() {
    const images = document.querySelectorAll('img[loading="lazy"]');

    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src || img.src;
                    img.classList.add('loaded');
                    observer.unobserve(img);
                }
            });
        });

        images.forEach(img => imageObserver.observe(img));
    }
}

// ============================================
// Mobile Menu Toggle
// ============================================
function toggleMobileMenu() {
    const navLinks = document.getElementById('nav-links');
    const menuBtn = document.querySelector('.mobile-menu-btn');
    navLinks.classList.toggle('active');
    menuBtn.classList.toggle('active');
}

function initMobileMenu() {
    // Close mobile menu when a nav link is clicked
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            const navLinks = document.getElementById('nav-links');
            const menuBtn = document.querySelector('.mobile-menu-btn');
            if (navLinks.classList.contains('active')) {
                navLinks.classList.remove('active');
                menuBtn.classList.remove('active');
            }
        });
    });
}

// ============================================
// Initialize All Functions
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    fadeInOnScroll();
    updateActiveNavLink();
    lazyLoadImages();
    initMobileMenu();
    initTripleCompare();
    initCompareCarousel();

    console.log('🚀 TrajVG website loaded successfully!');
});

// ============================================
// Triple Video Compare (Two Sliders)
// ============================================
function initTripleCompare() {
    const comps = document.querySelectorAll('[data-triple]');
    comps.forEach((comp) => {
        const minGap = Number(comp.dataset.minGap) || 8;
        let split1 = Number(comp.dataset.split1) || 33;
        let split2 = Number(comp.dataset.split2) || 66;

        const handle1 = comp.querySelector('[data-handle="1"]');
        const handle2 = comp.querySelector('[data-handle="2"]');
        const baseVideo = comp.querySelector('.tc-base-video');
        const overlayVideos = Array.from(comp.querySelectorAll('.tc-overlay-video'));

        const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

        const applySplits = (nextSplit1, nextSplit2) => {
            let s1 = clamp(nextSplit1, 0, 100);
            let s2 = clamp(nextSplit2, 0, 100);
            if (s2 - s1 < minGap) {
                if (s1 !== split1) {
                    s1 = s2 - minGap;
                } else {
                    s2 = s1 + minGap;
                }
            }
            s1 = clamp(s1, 0, 100 - minGap);
            s2 = clamp(s2, minGap, 100);
            split1 = s1;
            split2 = s2;
            comp.style.setProperty('--split1', `${split1}%`);
            comp.style.setProperty('--split2', `${split2}%`);
            if (handle1) {
                handle1.setAttribute('aria-valuenow', Math.round(split1).toString());
            }
            if (handle2) {
                handle2.setAttribute('aria-valuenow', Math.round(split2).toString());
            }
        };

        const getValueFromEvent = (event) => {
            const rect = comp.getBoundingClientRect();
            const x = clamp(event.clientX - rect.left, 0, rect.width);
            return (x / rect.width) * 100;
        };

        const startDrag = (which) => (event) => {
            if (event.pointerType === 'mouse' && event.button !== 0) return;
            event.preventDefault();
            const update = (e) => {
                const value = getValueFromEvent(e);
                if (which === 1) {
                    applySplits(value, split2);
                } else {
                    applySplits(split1, value);
                }
            };
            const stop = () => {
                window.removeEventListener('pointermove', update);
                window.removeEventListener('pointerup', stop);
            };
            window.addEventListener('pointermove', update);
            window.addEventListener('pointerup', stop, { once: true });
            update(event);
        };

        if (handle1) {
            handle1.addEventListener('pointerdown', startDrag(1));
            handle1.addEventListener('keydown', (event) => {
                const step = event.shiftKey ? 5 : 1;
                if (event.key === 'ArrowLeft') {
                    event.preventDefault();
                    applySplits(split1 - step, split2);
                }
                if (event.key === 'ArrowRight') {
                    event.preventDefault();
                    applySplits(split1 + step, split2);
                }
            });
        }

        if (handle2) {
            handle2.addEventListener('pointerdown', startDrag(2));
            handle2.addEventListener('keydown', (event) => {
                const step = event.shiftKey ? 5 : 1;
                if (event.key === 'ArrowLeft') {
                    event.preventDefault();
                    applySplits(split1, split2 - step);
                }
                if (event.key === 'ArrowRight') {
                    event.preventDefault();
                    applySplits(split1, split2 + step);
                }
            });
        }

        const divider1 = comp.querySelector('.tc-divider-1');
        const divider2 = comp.querySelector('.tc-divider-2');
        if (divider1) divider1.addEventListener('pointerdown', startDrag(1));
        if (divider2) divider2.addEventListener('pointerdown', startDrag(2));

        if (baseVideo && overlayVideos.length > 0) {
            const syncThreshold = 0.05;
            let syncing = false;
            let rafId = null;

            const syncToBase = () => {
                if (syncing) return;
                syncing = true;
                const t = baseVideo.currentTime;
                overlayVideos.forEach((video) => {
                    if (!Number.isFinite(video.duration)) {
                        return;
                    }
                    if (Math.abs(video.currentTime - t) > syncThreshold) {
                        try {
                            video.currentTime = t;
                        } catch (err) {
                            // Ignore sync errors on some browsers.
                        }
                    }
                    video.playbackRate = baseVideo.playbackRate;
                });
                syncing = false;
            };

            baseVideo.addEventListener('play', () => {
                overlayVideos.forEach((video) => {
                    if (video.paused) {
                        video.play().catch(() => {});
                    }
                });
                syncToBase();
                if (rafId) {
                    cancelAnimationFrame(rafId);
                }
                const tick = () => {
                    if (baseVideo.paused) return;
                    syncToBase();
                    rafId = requestAnimationFrame(tick);
                };
                rafId = requestAnimationFrame(tick);
            });
            baseVideo.addEventListener('pause', () => {
                overlayVideos.forEach((video) => video.pause());
                if (rafId) {
                    cancelAnimationFrame(rafId);
                    rafId = null;
                }
            });
            baseVideo.addEventListener('seeking', syncToBase);
            baseVideo.addEventListener('seeked', syncToBase);
            baseVideo.addEventListener('timeupdate', syncToBase);
            baseVideo.addEventListener('ratechange', syncToBase);
            baseVideo.addEventListener('loadedmetadata', syncToBase);
            overlayVideos.forEach((video) => {
                video.addEventListener('loadedmetadata', syncToBase);
            });
        }

        applySplits(split1, split2);
    });
}

// ============================================
// Compare Carousel Dots
// ============================================
function initCompareCarousel() {
    const carousels = document.querySelectorAll('[data-compare-carousel]');
    carousels.forEach((carousel) => {
        const items = Array.from(carousel.querySelectorAll('.compare-item'));
        const dots = carousel.querySelector('[data-compare-dots]');
        if (!dots || items.length === 0) {
            return;
        }

        dots.innerHTML = '';
        let activeIndex = items.findIndex((item) => item.classList.contains('is-active'));
        if (activeIndex < 0) {
            activeIndex = 0;
        }

        const setActive = (index) => {
            const nextIndex = Math.max(0, Math.min(items.length - 1, index));
            items.forEach((item, idx) => {
                const isActive = idx === nextIndex;
                item.classList.toggle('is-active', isActive);
                item.setAttribute('aria-hidden', isActive ? 'false' : 'true');
                item.querySelectorAll('video').forEach((video) => {
                    if (isActive) {
                        if (video.autoplay) {
                            video.play().catch(() => {});
                        }
                    } else {
                        video.pause();
                    }
                });
            });
            dots.querySelectorAll('.compare-dot').forEach((dot, idx) => {
                const isActive = idx === nextIndex;
                dot.classList.toggle('is-active', isActive);
                dot.setAttribute('aria-current', isActive ? 'true' : 'false');
            });
            activeIndex = nextIndex;
        };

        items.forEach((item, idx) => {
            const dot = document.createElement('button');
            dot.type = 'button';
            dot.className = 'compare-dot';
            dot.setAttribute('aria-label', `Switch to comparison ${idx + 1}`);
            dot.addEventListener('click', () => setActive(idx));
            dots.appendChild(dot);
        });

        let startX = 0;
        let startY = 0;
        let isDragging = false;
        const swipeThreshold = 40;

        const shouldIgnoreSwipe = (target) => {
            if (!(target instanceof Element)) return false;
            return Boolean(target.closest('.tc-handle, .tc-divider, .compare-dot'));
        };

        const onPointerDown = (event) => {
            if (event.pointerType === 'mouse' && event.button !== 0) return;
            if (shouldIgnoreSwipe(event.target)) return;
            isDragging = true;
            startX = event.clientX;
            startY = event.clientY;
            if (carousel.setPointerCapture) {
                carousel.setPointerCapture(event.pointerId);
            }
        };

        const onPointerUp = (event) => {
            if (!isDragging) return;
            isDragging = false;
            const dx = event.clientX - startX;
            const dy = event.clientY - startY;
            if (Math.abs(dx) > Math.max(swipeThreshold, Math.abs(dy) * 1.2)) {
                if (dx < 0) {
                    setActive((activeIndex + 1) % items.length);
                } else {
                    setActive((activeIndex - 1 + items.length) % items.length);
                }
            }
        };

        carousel.addEventListener('pointerdown', onPointerDown);
        carousel.addEventListener('pointerup', onPointerUp);
        carousel.addEventListener('pointercancel', () => {
            isDragging = false;
        });

        carousel.addEventListener('click', (event) => {
            if (shouldIgnoreSwipe(event.target)) return;
            const rect = carousel.getBoundingClientRect();
            const x = event.clientX - rect.left;
            if (x < rect.width / 2) {
                setActive((activeIndex - 1 + items.length) % items.length);
            } else {
                setActive((activeIndex + 1) % items.length);
            }
        });

        setActive(activeIndex);
    });
}

// ============================================
// Performance Optimization
// ============================================
// Debounce function for scroll events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Apply debounce to scroll-heavy functions
const debouncedScroll = debounce(() => {
    // Any heavy scroll operations can go here
}, 100);

window.addEventListener('scroll', debouncedScroll);
